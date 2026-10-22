# SPDX-License-Identifier: Apache-2.0
"""Sorted MoE prefill gathers that skip the padding rows of partial tiles.

MLX runs sorted ``gather_qmm`` prefill batches through ``affine_gather_qmm_rhs``
with one fixed 16x32 tile. Top-k routing leaves every expert with a partly
filled last row tile (tens of rows per expert in a 4096-token chunk for
128-512 expert models), and the stock kernel spends its MMAs on the padding
rows of those tiles.

This is the same MLX kernel, compiled from MLX's own Metal headers with a
64x32 tile. Partial row tiles are specialized on the number of 8-row MMA
fragments that hold a valid row and skip the others, so the larger tile costs
nothing on small experts. Dequantization
and each valid row's MMA sequence (8-wide K steps in order, fp32 accumulators)
are MLX's, so the output is bit-identical to ``mx.gather_qmm``; every
shape/tile pair is also compared with ``mx.gather_qmm`` on its first batch and
falls back to MLX for the rest of the process on any difference.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# (BM, BN, BK, WM, WN). M3 Ultra, Qwen MoE expert shapes (128-512 experts,
# 4/8-bit, 1K-4K token chunks): never slower than MLX's 16x32, 1.04-1.23x.
_TILE = (64, 32, 32, 2, 2)

_DISABLED = os.environ.get("OMLX_GATHER_QMM_TILES", "1") == "0"
_KERNELS: tuple[Any, Any] | None = None
_VALIDATED: set[tuple] = set()
_REJECTED: set[tuple] = set()
_LOGGED = False

_STEEL_HEADERS = (
    "steel/defines.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/utils.h",
    "steel/gemm/transforms.h",
    "steel/gemm/loader.h",
    "steel/gemm/mma.h",
    "steel/gemm/params.h",
    "steel/gemm/gemm.h",
    "quantized_utils.h",
)

# MLX's affine_gather_qmm_rhs (quantized.h) as a function, with the
# partial-row-tile branches replaced by _partial_tile_loop below.
_IMPL = r"""
template <typename T, typename mma_t, int BK, int WM, int WN, short n_frags>
METAL_FUNC void omlx_tiles_mma(
    thread mma_t& op, const threadgroup T* As, const threadgroup T* Bs) {
  // BlockMMA::mma restricted to the first n_frags row fragments.
  using Frag = typename mma_t::MMAFrag_acc_t;
  constexpr short TN = mma_t::TN;
  constexpr short A_str_m = mma_t::A_str_m;
  constexpr short A_str_k = mma_t::A_str_k;
  constexpr short B_str_k = mma_t::B_str_k;
  constexpr short B_str_n = mma_t::B_str_n;
  As += op.As_offset;
  Bs += op.Bs_offset;
  STEEL_PRAGMA_UNROLL
  for (short kk = 0; kk < BK; kk += 8) {
    simdgroup_barrier(mem_flags::mem_none);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_frags; ++i) {
      Frag::load(op.Atile.frag_at(i, 0), &(As[(i * 8) * WM * A_str_m]),
                 mlx::steel::Int<A_str_m>{}, mlx::steel::Int<A_str_k>{});
    }
    simdgroup_barrier(mem_flags::mem_none);
    op.Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);
    simdgroup_barrier(mem_flags::mem_none);
    STEEL_PRAGMA_UNROLL
    for (short m = 0; m < n_frags; ++m) {
      STEEL_PRAGMA_UNROLL
      for (short n = 0; n < TN; ++n) {
        short n_serp = (m % 2) ? (TN - 1 - n) : n;
        Frag::mma(op.Ctile.frag_at(m, n_serp), op.Atile.frag_at(m, 0),
                  op.Btile.frag_at(0, n_serp), op.Ctile.frag_at(m, n_serp));
      }
    }
    As += 8 * A_str_k;
    Bs += 8 * B_str_k;
  }
}

template <typename T, typename mma_t, typename loader_x_t, typename loader_w_t,
          int BK, int WM, int WN, short n_frags, bool align_N, bool align_K>
METAL_FUNC void omlx_tiles_partial_loop(
    thread mma_t& mma_op, thread loader_x_t& loader_x, thread loader_w_t& loader_w,
    threadgroup T* Xs, threadgroup T* Ws, const int K_it, const short tgp_bm,
    const short tgp_bn, const short BN, const short2 tile_x, const short2 tile_w) {
  if (n_frags > mma_t::TM) {
    return;
  }
  constexpr short frags = n_frags > mma_t::TM ? mma_t::TM : n_frags;
  for (int k = 0; k < K_it; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(BK, tgp_bm));
    if (align_N || tgp_bn == BN) {
      loader_w.load_unsafe();
    } else {
      loader_w.load_safe(short2(BK, tgp_bn));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    omlx_tiles_mma<T, mma_t, BK, WM, WN, frags>(mma_op, Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  if (!align_K) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(tile_x);
    loader_w.load_safe(tile_w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    omlx_tiles_mma<T, mma_t, BK, WM, WN, frags>(mma_op, Xs, Ws);
  }
}

template <typename T, int group_size, int bits, int BM, int BN, int BK, int WM,
          int WN, bool align_N, bool align_K>
METAL_FUNC void omlx_tiles_gather_qmm_rhs(
    const device T* x, const device uint32_t* w, const device T* scales,
    const device T* biases, const device int32_t* offsets, device T* y,
    const int M, const int N, const int K, const int num_groups,
    threadgroup T* Xs, threadgroup T* Ws,
    uint3 tid, uint simd_group_id, uint simd_lane_id) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true,
                                     BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, true,
                                          WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = N * K_w;
  const size_t stride_s = N * K_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(offsets, num_groups, M, tid.y, simd_lane_id,
                             y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += y_col_long * K_w;
  scales += y_col_long * K_g;
  biases += y_col_long * K_g;
  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl + group * stride_w, scales + group * stride_s,
                             biases + group * stride_s, K, Ws, simd_group_id,
                             simd_lane_id);
  if (tgp_bm == BM && (align_N || tgp_bn == BN)) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(y, N);
    return;
  }
  // This simdgroup's row fragment i holds tile rows tm + i * 8 * WM .. + 7.
  const short tm = 8 * (simd_group_id / WN);
  const short n_frags = short(clamp((int(tgp_bm) - tm + 8 * WM - 1) / (8 * WM),
                                    0, int(mma_t::TM)));
  switch (n_frags) {
#define OMLX_TILES_CASE(F)                                                   \
    case F:                                                                  \
      omlx_tiles_partial_loop<T, mma_t, loader_x_t, loader_w_t, BK, WM, WN, \
                              F, align_N, align_K>(                          \
          mma_op, loader_x, loader_w, Xs, Ws, K_it, tgp_bm, tgp_bn, BN,      \
          tile_x, tile_w);                                                   \
      break;
    OMLX_TILES_CASE(0)
    OMLX_TILES_CASE(1)
    OMLX_TILES_CASE(2)
    OMLX_TILES_CASE(3)
    OMLX_TILES_CASE(4)
    OMLX_TILES_CASE(5)
    OMLX_TILES_CASE(6)
    OMLX_TILES_CASE(7)
    OMLX_TILES_CASE(8)
#undef OMLX_TILES_CASE
  }
  mma_op.store_result_safe(y, N, short2(tgp_bn, tgp_bm));
}
"""

_BODY = r"""
  constexpr int BKP = BK + 16 / sizeof(T);
  threadgroup T Xs[BM * BKP];
  threadgroup T Ws[BN * BKP];
  omlx_tiles_gather_qmm_rhs<T, GS, BITS, BM, BN, BK, WM, WN, ALIGN_N, ALIGN_K>(
      x, w, scales, biases, offsets, y, dims[0], dims[1], dims[2], dims[3],
      Xs, Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
"""

# MLX's gather_mm_offsets: the first sorted row of each expert.
_OFFSETS_BODY = r"""
  uint g = thread_position_in_grid.x;
  if (g >= (uint)E) {
    return;
  }
  int lo = 0;
  int hi = rows[0];
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (indices[mid] < g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  offsets[g] = lo;
"""


def _strip_includes(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("#pragma once")
        and not re.match(r'#include\s+"mlx/', line)
    )


def _header() -> str:
    """MLX's steel GEMM and quantized block loader, from the installed MLX."""
    root = Path(mx.__file__).parent / "include" / "mlx" / "backend" / "metal"
    root = root / "kernels"
    parts = [_strip_includes((root / name).read_text()) for name in _STEEL_HEADERS]
    lines = (root / "quantized.h").read_text().splitlines()
    # Everything before the first matmul implementation: the dequantize
    # helpers and QuantizedBlockLoader, minus the function constants that
    # only MLX's own kernels specialize.
    end = next(i for i, line in enumerate(lines) if "qmv_quad_impl(" in line)
    while not lines[end].startswith("template"):
        end -= 1
    prefix = "\n".join(line for line in lines[:end] if "function_constant" not in line)
    return "\n".join(parts) + "\n" + _strip_includes(prefix) + "\n" + _IMPL


def _kernels() -> tuple[Any, Any]:
    global _KERNELS
    if _KERNELS is None:
        gather = mx.fast.metal_kernel(
            name="omlx_tiled_gather_qmm_rhs",
            input_names=["x", "w", "scales", "biases", "offsets", "dims"],
            output_names=["y"],
            source=_BODY,
            header=_header(),
            ensure_row_contiguous=True,
        )
        offsets = mx.fast.metal_kernel(
            name="omlx_tiled_gather_offsets",
            input_names=["indices", "rows"],
            output_names=["offsets"],
            source=_OFFSETS_BODY,
            ensure_row_contiguous=True,
        )
        _KERNELS = (gather, offsets)
    return _KERNELS


def _tiled(linear: Any, x: mx.array, indices: mx.array, tile: tuple) -> mx.array:
    gather, offsets_kernel = _kernels()
    bm, bn, bk, wm, wn = tile
    n_experts, n_out = linear.weight.shape[:2]
    k_in = x.shape[-1]
    rows = x.size // k_in
    flat = indices.reshape(rows).astype(mx.uint32)
    offsets = offsets_kernel(
        inputs=[flat, mx.array([rows], dtype=mx.int32)],
        template=[("E", n_experts)],
        grid=(n_experts, 1, 1),
        threadgroup=(min(n_experts, 256), 1, 1),
        output_shapes=[(n_experts,)],
        output_dtypes=[mx.int32],
    )[0]
    tiles_n = (n_out + bn - 1) // bn
    # MLX's grid: enough row tiles for every expert's partial last tile.
    tiles_m = min(rows, (rows + bm - 1) // bm + n_experts - 1)
    y = gather(
        inputs=[
            x.reshape(rows, k_in),
            linear.weight,
            linear.scales,
            linear.biases,
            offsets,
            mx.array([rows, n_out, k_in, n_experts], dtype=mx.int32),
        ],
        template=[
            ("T", x.dtype),
            ("GS", linear.group_size),
            ("BITS", linear.bits),
            ("BM", bm),
            ("BN", bn),
            ("BK", bk),
            ("WM", wm),
            ("WN", wn),
            ("ALIGN_N", n_out % bn == 0),
            ("ALIGN_K", k_in % bk == 0),
        ],
        grid=(32 * tiles_n, wn * tiles_m, wm),
        threadgroup=(32, wn, wm),
        output_shapes=[(rows, n_out)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(*x.shape[:-1], n_out)


def _eligible_tile(linear: Any, x: mx.array, indices: mx.array) -> tuple | None:
    if _DISABLED or not mx.metal.is_available():
        return None
    if getattr(linear, "mode", "affine") != "affine" or "bias" in linear:
        return None
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None or weight.ndim != 3:
        return None
    if x.dtype not in (mx.float16, mx.bfloat16) or x.ndim < 2 or x.shape[-2] != 1:
        return None
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        return None
    n_experts = weight.shape[0]
    k_in = x.shape[-1]
    if weight.shape[2] * 32 != k_in * linear.bits:
        return None
    rows = x.size // k_in
    # MLX takes gather_qmm_rhs only for these batches; anything else keeps
    # MLX's kernel (and its numerics).
    if indices.size != rows or rows < 16 or rows // n_experts < 4:
        return None
    from omlx.custom_kernels.nax import is_nax_available

    if is_nax_available():
        return None
    return _TILE


def tiled_gather_qmm(linear: Any, x: mx.array, indices: mx.array) -> mx.array | None:
    """``linear(x, indices, sorted_indices=True)`` via the tiled kernel, or None.

    ``x`` is ``[rows, 1, K]`` with rows sorted by expert and ``indices`` the
    matching expert ids. None means the caller keeps MLX's kernel.
    """
    global _DISABLED, _LOGGED
    tile = _eligible_tile(linear, x, indices)
    if tile is None:
        return None
    signature = (
        tuple(linear.weight.shape),
        x.shape[-1],
        linear.bits,
        linear.group_size,
        x.dtype,
        tile,
    )
    if signature in _REJECTED:
        return None
    try:
        y = _tiled(linear, x, indices, tile)
        if signature not in _VALIDATED:
            reference = mx.gather_qmm(
                x,
                linear.weight,
                linear.scales,
                linear.biases,
                rhs_indices=indices,
                transpose=True,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
                sorted_indices=True,
            )
            if not mx.array_equal(y, reference).item():
                _REJECTED.add(signature)
                logger.warning(
                    "Tiled gather_qmm differs from MLX for %s; keeping MLX's kernel",
                    signature,
                )
                return reference
            _VALIDATED.add(signature)
        if not _LOGGED:
            _LOGGED = True
            logger.info("Tiled sorted gather_qmm engaged for MoE prefill")
        return y
    except Exception:  # noqa: BLE001 - optional fast path
        _DISABLED = True
        logger.warning("Tiled gather_qmm failed; using MLX's kernel", exc_info=True)
        return None


__all__ = ["tiled_gather_qmm"]
