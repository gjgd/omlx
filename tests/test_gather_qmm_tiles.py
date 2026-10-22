# SPDX-License-Identifier: Apache-2.0
"""Tests for the partial-tile sorted MoE gather_qmm prefill kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import gather_qmm_tiles as tiles


def _nax() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return is_nax_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or _nax(),
    reason="non-NAX Metal GPU required",
)


def _switch_linear(n_experts, n_in, n_out, bits, dtype=mx.bfloat16):
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    linear = QuantizedSwitchLinear(
        n_in, n_out, n_experts, bias=False, group_size=64, bits=bits
    )
    w = mx.random.normal((n_experts, n_out, n_in)) * 0.05
    linear.weight, scales, biases = mx.quantize(w, group_size=64, bits=bits)
    linear.scales = scales.astype(dtype)
    linear.biases = biases.astype(dtype)
    mx.eval(linear.parameters())
    return linear


def _sorted_routes(n_experts, tokens, top_k, seed=0):
    # Skewed popularity so experts get full tiles, partial tiles and no rows.
    rng = np.random.default_rng(seed)
    logits = np.log(rng.lognormal(0, 1.0, n_experts))[None]
    logits = logits + rng.gumbel(size=(tokens, n_experts))
    idx = np.argpartition(-logits, top_k, axis=1)[:, :top_k].reshape(-1)
    return mx.array(np.sort(idx).astype(np.uint32))


def _reference(linear, x, idx):
    return mx.gather_qmm(
        x,
        linear.weight,
        linear.scales,
        linear.biases,
        rhs_indices=idx,
        transpose=True,
        group_size=linear.group_size,
        bits=linear.bits,
        sorted_indices=True,
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "n_experts,n_in,n_out", [(32, 256, 192), (16, 128, 80), (64, 192, 256)]
)
def test_tiled_kernel_matches_mlx_bitwise(bits, n_experts, n_in, n_out):
    mx.random.seed(0)
    linear = _switch_linear(n_experts, n_in, n_out, bits)
    idx = _sorted_routes(n_experts, tokens=512, top_k=4)
    x = (mx.random.normal((idx.size, 1, n_in)) * 0.5).astype(mx.bfloat16)

    got = tiles._tiled(linear, x, idx, tiles._TILE)
    want = _reference(linear, x, idx)

    assert got.shape == want.shape
    assert mx.array_equal(got, want).item()


def test_tiled_gather_qmm_engages_and_matches(monkeypatch):
    monkeypatch.setattr(tiles, "_VALIDATED", set())
    monkeypatch.setattr(tiles, "_REJECTED", set())
    mx.random.seed(1)
    linear = _switch_linear(32, 256, 128, 8)
    idx = _sorted_routes(32, tokens=256, top_k=4, seed=1)
    x = (mx.random.normal((idx.size, 1, 256)) * 0.5).astype(mx.bfloat16)

    got = tiles.tiled_gather_qmm(linear, x, idx)

    assert got is not None
    assert mx.array_equal(got, _reference(linear, x, idx)).item()
    assert len(tiles._VALIDATED) == 1


def test_small_batches_keep_mlx_kernel():
    linear = _switch_linear(32, 256, 128, 8)
    # Fewer than 4 rows per expert: MLX does not use gather_qmm_rhs here.
    idx = _sorted_routes(32, tokens=16, top_k=4)
    x = mx.zeros((idx.size, 1, 256), dtype=mx.bfloat16)

    assert tiles.tiled_gather_qmm(linear, x, idx) is None
