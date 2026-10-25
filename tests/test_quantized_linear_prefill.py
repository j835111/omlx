# SPDX-License-Identifier: Apache-2.0
"""Tests for structural affine q4 prefill linear routing."""

from __future__ import annotations

from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.quantized_linear_prefill as patch


@pytest.fixture(autouse=True)
def native_kernel():
    if not mx.metal.is_available():
        pytest.skip("Metal is required")
    if not patch._has_native_qmm():
        pytest.skip("The native affine q4 qmm kernel is required")
    patch._PROBE_CACHE.clear()
    yield
    patch._PROBE_CACHE.clear()


def _model(*, bias=False, bits=4, group_size=64, output_dim=192):
    model = nn.Module()
    model.proj = nn.Linear(256, output_dim, bias=bias)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=group_size, bits=bits, mode="affine")
    model.eval()
    return model


def _inputs(tokens):
    return mx.random.normal((1, tokens, 256), key=mx.random.key(7)).astype(mx.bfloat16)


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert mx.allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=2e-2, atol=2e-3
    ).item()


def test_structural_swap_is_instance_local_cached_and_idempotent(monkeypatch):
    model = _model()
    model.other = _model().proj
    untouched = _model().proj
    probe = Mock(wraps=patch._probe)
    monkeypatch.setattr(patch, "_probe", probe)

    assert patch.apply_quantized_linear_prefill(model) == 2
    assert isinstance(model.proj, nn.QuantizedLinear)
    assert type(model.proj) is type(model.other)
    assert type(model.proj) is not nn.QuantizedLinear
    assert type(model.proj)._omlx_q4_prefill is True
    assert type(model.proj)._omlx_q4_prefill_failed is False
    assert type(untouched) is nn.QuantizedLinear
    assert not getattr(nn.QuantizedLinear, "_omlx_q4_prefill", False)
    assert patch.apply_quantized_linear_prefill(model) == 0
    probe.assert_called_once()


@pytest.mark.parametrize("kwargs", [{"bias": True}, {"bits": 8}, {"output_dim": 160}])
def test_ineligible_linears_are_not_swapped(kwargs):
    model = _model(**kwargs)
    assert patch.apply_quantized_linear_prefill(model) == 0
    assert type(model.proj) is nn.QuantizedLinear


def test_other_class_is_not_swapped():
    class CustomLinear(nn.QuantizedLinear):
        pass

    model = _model()
    model.proj.__class__ = CustomLinear
    model.dense = nn.Linear(256, 192, bias=False)
    assert patch.apply_quantized_linear_prefill(model) == 0
    assert type(model.proj) is CustomLinear
    assert type(model.dense) is nn.Linear


@pytest.mark.parametrize("name", ["lm_head", "embed_tokens", "embed_tokens_per_layer"])
@pytest.mark.parametrize("nested", [False, True])
def test_output_and_embedding_names_are_not_swapped(name, nested):
    model = nn.Module()
    parent = model
    if nested:
        model.language_model = nn.Module()
        parent = model.language_model
    parent[name] = _model().proj
    assert patch.apply_quantized_linear_prefill(model) == 0
    assert type(parent[name]) is nn.QuantizedLinear


@pytest.mark.parametrize("tokens", [1, patch._MIN_TOKENS - 1])
def test_short_prefill_uses_original(monkeypatch, tokens):
    model = _model()
    x = _inputs(tokens)
    expected = model.proj(x)
    mx.eval(expected)
    assert patch.apply_quantized_linear_prefill(model) == 1
    fast = Mock(side_effect=AssertionError("short chunks must use stock"))
    gate = Mock(side_effect=AssertionError("short chunks must exit immediately"))
    monkeypatch.setattr(patch, "_linear_qmm", fast)
    monkeypatch.setattr(patch, "_can_route_affine_linear", gate)

    assert mx.array_equal(model.proj(x), expected).item()
    fast.assert_not_called()
    gate.assert_not_called()


@pytest.mark.parametrize("tokens", [patch._MIN_TOKENS, patch._MIN_TOKENS + 1])
def test_long_prefill_routes_and_matches_stock(monkeypatch, tokens):
    model = _model()
    x = _inputs(tokens)
    expected = model.proj(x)
    mx.eval(expected)
    assert patch.apply_quantized_linear_prefill(model) == 1
    fast = Mock(wraps=patch._linear_qmm)
    monkeypatch.setattr(patch, "_linear_qmm", fast)

    _assert_close(model.proj(x), expected)
    fast.assert_called_once_with(model.proj, x, patch._QMM_VARIANT)


def test_kernel_failure_warns_once_and_falls_back(monkeypatch, caplog):
    model = _model()
    x = _inputs(patch._MIN_TOKENS)
    expected = model.proj(x)
    mx.eval(expected)
    assert patch.apply_quantized_linear_prefill(model) == 1
    fast = Mock(side_effect=RuntimeError("test kernel failure"))
    monkeypatch.setattr(patch, "_linear_qmm", fast)

    for _ in range(2):
        assert mx.array_equal(model.proj(x), expected).item()
    fast.assert_called_once()
    assert model.proj._omlx_q4_prefill_failed is True
    warnings = [r for r in caplog.records if "fast path failed" in r.message]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_probe_rejection_is_cached_and_uses_original_class(monkeypatch):
    model = _model()
    other = _model()

    def garbage(module, x, variant):
        assert type(module) is nn.QuantizedLinear
        assert x.shape == (1, 64, 256)
        assert x.dtype == mx.bfloat16
        return mx.full((1, 64, 192), 1e4, dtype=x.dtype)

    fast = Mock(side_effect=garbage)
    stock = nn.QuantizedLinear.__call__
    original = Mock(side_effect=stock)
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", original)
    monkeypatch.setattr(patch, "_linear_qmm", fast)

    assert patch.apply_quantized_linear_prefill(model) == 0
    assert patch.apply_quantized_linear_prefill(other) == 0
    assert type(model.proj) is nn.QuantizedLinear
    original.assert_called_once()
    fast.assert_called_once()


@pytest.mark.parametrize("missing", ["metal", "native"])
def test_unavailable_backend_rewrites_nothing(monkeypatch, missing):
    model = _model()
    if missing == "metal":
        monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    else:
        monkeypatch.setattr(patch, "_has_native_qmm", lambda: False)
    assert patch.apply_quantized_linear_prefill(model) == 0
    assert type(model.proj) is nn.QuantizedLinear


@pytest.mark.parametrize(
    "module_name", ["mlx_vlm.models.qwen3_5_moe.language", "qwen4_exp.language"]
)
def test_dedicated_family_models_are_skipped(module_name):
    model = _model()
    marker = type("Qwen", (nn.Module,), {"__module__": module_name})()
    model.family = marker
    assert patch.apply_quantized_linear_prefill(model) == 0
    assert type(model.proj) is nn.QuantizedLinear
