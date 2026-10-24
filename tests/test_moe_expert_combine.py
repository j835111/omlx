# SPDX-License-Identifier: Apache-2.0
"""Tests for structural MoE expert-combine routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

import omlx.patches.moe_expert_combine as patch


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.switch_glu = SwitchGLU(64, 32, 16)
        # fp32 expert weights are never routed (the kernel needs fp16/bf16).
        self.set_dtype(mx.bfloat16)

    def __call__(self, x, top_k_indices, top_k_weights):
        w = mx.expand_dims(top_k_weights, -1)
        y = self.switch_glu(x, top_k_indices)
        return (w * y).sum(-2)


class SharedExperts(Experts):
    def __init__(self):
        super().__init__()
        self.shared_expert = nn.Linear(64, 64)


class DifferentArity(Experts):
    def __call__(self, x, indices):
        return self.switch_glu(x, indices)


class ScaledExperts(Experts):
    def __call__(self, x, indices, weights):
        return 2 * super().__call__(x, indices, weights)


@pytest.fixture(autouse=True)
def clear_probe_cache():
    patch._PROBE_CACHE.clear()
    yield
    patch._PROBE_CACHE.clear()


@pytest.fixture
def available_kernel(monkeypatch):
    kernel = Mock()
    monkeypatch.setattr(patch, "_native_weighted_sum", lambda: kernel)
    monkeypatch.setattr(patch, "_probe", Mock(return_value=True))
    return kernel


@pytest.fixture
def native_kernel():
    if not mx.metal.is_available():
        pytest.skip("Metal is required")
    kernel = patch._native_weighted_sum()
    if kernel is None:
        pytest.skip("The native moe_weighted_sum kernel is required")
    return kernel


def _inputs(tokens, dtype=mx.bfloat16, weights_dtype=None, top_k=8):
    x = mx.random.normal((1, tokens, 64), key=mx.random.key(7)).astype(dtype)
    logits = mx.random.normal((1, tokens, 16), key=mx.random.key(8))
    indices = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
    weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
    return x, indices, weights.astype(weights_dtype or dtype)


def _experts(dtype=mx.bfloat16, cls=Experts):
    experts = cls()
    experts.set_dtype(dtype)
    experts.eval()
    return experts


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
    tolerance = 0.05 * mx.max(mx.abs(expected.astype(mx.float32))) + 1e-3
    assert (error <= tolerance).item()


def test_structural_match_is_instance_local_and_cached(available_kernel):
    model = nn.Module()
    model.layers = [Experts(), Experts()]
    untouched = Experts()

    assert patch.apply_moe_expert_combine(model) == 2
    first, second = model.layers
    assert type(first) is type(second)
    assert isinstance(first, Experts)
    assert type(first).__name__ == Experts.__name__
    assert type(first).__module__ == Experts.__module__
    assert type(first).__qualname__ == Experts.__qualname__
    assert type(first)._omlx_expert_combine
    assert type(untouched) is Experts
    assert not getattr(Experts, "_omlx_expert_combine", False)


def test_probe_cache_reuses_class_pair(available_kernel):
    assert patch.apply_moe_expert_combine(Experts()) == 1
    assert patch.apply_moe_expert_combine(Experts()) == 1
    patch._probe.assert_called_once()


def test_failed_probe_is_cached(available_kernel):
    patch._probe.return_value = False
    assert patch.apply_moe_expert_combine(Experts()) == 0
    assert patch.apply_moe_expert_combine(Experts()) == 0
    patch._probe.assert_called_once()


@pytest.mark.parametrize("wrapper", [SharedExperts, DifferentArity])
def test_structural_mismatch_is_not_rewritten(available_kernel, wrapper):
    experts = wrapper()
    assert patch.apply_moe_expert_combine(experts) == 0
    assert type(experts) is wrapper
    patch._probe.assert_not_called()


@pytest.mark.parametrize("extra", ["parameter", "child", "scalar", "all"])
def test_extra_state_is_not_rewritten(available_kernel, extra):
    experts = Experts()
    if extra in ("parameter", "all"):
        experts.scale = mx.array(2.0)
    if extra in ("child", "all"):
        experts.act = nn.GELU()
    if extra in ("scalar", "all"):
        experts.k = 8
    assert patch.apply_moe_expert_combine(experts) == 0
    patch._probe.assert_not_called()


def test_apply_is_idempotent(available_kernel):
    experts = Experts()
    assert patch.apply_moe_expert_combine(experts) == 1
    assert patch.apply_moe_expert_combine(experts) == 0


@pytest.mark.parametrize("override", [False, True])
def test_switch_glu_subclass_matches(available_kernel, override):
    class CustomSwitchGLU(SwitchGLU):
        pass

    if override:

        def custom_call(self, x, indices):
            return 2 * super(CustomSwitchGLU, self).__call__(x, indices)

        CustomSwitchGLU.__call__ = custom_call
    experts = Experts()
    experts.switch_glu = CustomSwitchGLU(64, 32, 16)
    experts.set_dtype(mx.bfloat16)
    assert patch.apply_moe_expert_combine(experts) == (0 if override else 1)


def test_native_unavailable_rewrites_nothing(monkeypatch):
    monkeypatch.setattr(patch, "_native_weighted_sum", lambda: None)
    experts = Experts()
    assert patch.apply_moe_expert_combine(experts) == 0
    assert type(experts) is Experts


@pytest.mark.parametrize("tokens", [1, patch._MIN_TOKENS - 1])
def test_short_chunks_use_original(available_kernel, monkeypatch, tokens):
    experts = _experts()
    inputs = _inputs(tokens)
    expected = experts(*inputs)
    assert patch.apply_moe_expert_combine(experts) == 1
    fast = Mock(side_effect=AssertionError("short chunks must use the original"))
    monkeypatch.setattr(patch, "_native_switch_weighted_sum", fast)
    lookup = Mock(side_effect=AssertionError("short chunks must exit immediately"))
    monkeypatch.setattr(patch, "_native_weighted_sum", lookup)

    actual = experts(*inputs)
    assert mx.array_equal(actual, expected).item()
    fast.assert_not_called()
    lookup.assert_not_called()


def test_fast_path_failure_falls_back_once(available_kernel, monkeypatch, caplog):
    experts = _experts()
    inputs = _inputs(patch._MIN_TOKENS)
    expected = experts(*inputs)
    assert patch.apply_moe_expert_combine(experts) == 1
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    fast = Mock(side_effect=RuntimeError("test kernel failure"))
    monkeypatch.setattr(patch, "_native_switch_weighted_sum", fast)

    for _ in range(2):
        assert mx.array_equal(experts(*inputs), expected).item()
    fast.assert_called_once()
    assert experts._omlx_expert_combine_failed is True
    warnings = [r for r in caplog.records if "fast path failed" in r.message]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_fp32_weights_use_original(available_kernel, monkeypatch):
    experts = _experts()
    inputs = _inputs(patch._MIN_TOKENS, weights_dtype=mx.float32)
    expected = experts(*inputs)
    assert patch.apply_moe_expert_combine(experts) == 1
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    fast = Mock(side_effect=AssertionError("mixed dtypes must use the original"))
    monkeypatch.setattr(patch, "_native_switch_weighted_sum", fast)
    actual = experts(*inputs)
    assert actual.dtype == mx.float32
    assert mx.array_equal(actual, expected).item()
    fast.assert_not_called()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_native_output_matches_stock(native_kernel, monkeypatch, dtype):
    experts = _experts(dtype)
    inputs = _inputs(patch._MIN_TOKENS, dtype)
    expected = experts(*inputs)
    mx.eval(expected)
    assert patch.apply_moe_expert_combine(experts) == 1
    fast = Mock(wraps=patch._native_switch_weighted_sum)
    monkeypatch.setattr(patch, "_native_switch_weighted_sum", fast)

    _assert_close(experts(*inputs), expected)
    fast.assert_called_once()


@pytest.mark.parametrize(
    "module_name",
    ["mlx_vlm.models.gemma4.language", "mlx_lm.models.gemma4_text"],
)
def test_real_gemma_experts(native_kernel, monkeypatch, module_name):
    module = pytest.importorskip(module_name)
    config = SimpleNamespace(hidden_size=64, moe_intermediate_size=32, num_experts=16)
    experts = module.Experts(config)
    experts.set_dtype(mx.bfloat16)
    experts.eval()
    inputs = _inputs(patch._MIN_TOKENS)
    expected = experts(*inputs)
    mx.eval(expected)
    assert patch.apply_moe_expert_combine(experts) == 1
    fast = Mock(wraps=patch._native_switch_weighted_sum)
    monkeypatch.setattr(patch, "_native_switch_weighted_sum", fast)
    _assert_close(experts(*inputs), expected)
    fast.assert_called_once()


def test_probe_rejects_extra_scaling(native_kernel, monkeypatch, caplog):
    experts = _experts(cls=ScaledExperts)
    monkeypatch.setattr(
        patch, "_matches_structure", lambda module, _: isinstance(module, ScaledExperts)
    )
    with caplog.at_level("DEBUG", logger=patch.__name__):
        assert patch.apply_moe_expert_combine(experts) == 0
    assert "numerical mismatch" in caplog.text
    assert type(experts) is ScaledExperts


def test_probe_preserves_state_and_rng(native_kernel):
    experts = _experts()
    keys = list(experts.keys())
    attrs = dict(vars(experts))
    parameters = experts.parameters()
    mx.random.seed(19)
    expected = mx.random.normal((8,))
    mx.eval(expected)
    mx.random.seed(19)
    assert patch._probe(experts, native_kernel)
    actual = mx.random.normal((8,))
    assert mx.array_equal(actual, expected).item()
    assert list(experts.keys()) == keys
    assert vars(experts) == attrs
    for name, projection in parameters["switch_glu"].items():
        for key, value in projection.items():
            assert experts.parameters()["switch_glu"][name][key] is value


def test_probe_requires_metal(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    original = Mock(side_effect=AssertionError("must not call original"))
    monkeypatch.setattr(Experts, "__call__", original)
    assert not patch._probe(Experts(), Mock())
    original.assert_not_called()


@pytest.mark.parametrize("failure", ["exception", "shape", "dtype", "nan"])
def test_probe_rejects_bad_native_output(monkeypatch, failure):
    experts = _experts()
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)

    def native(switch, x, indices, weights, kernel):
        if failure == "exception":
            raise RuntimeError("probe failure")
        if failure == "shape":
            return mx.zeros((1, 64, 1), dtype=x.dtype)
        if failure == "dtype":
            return mx.zeros(x.shape, dtype=mx.float32)
        return mx.full(x.shape, float("nan"), dtype=x.dtype)

    monkeypatch.setattr(patch, "_native_switch_weighted_sum", native)
    assert not patch._probe(experts, Mock())


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "short",
        "dtype",
        "weights_dtype",
        "shapes",
        "prefix",
        "top_k",
        "training",
        "sharding",
    ],
)
def test_route_guards(monkeypatch, case):
    experts = _experts()
    x, indices, weights = _inputs(patch._MIN_TOKENS)
    switch = experts.switch_glu
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    assert patch._should_route(switch, x, indices, weights, patch._MIN_TOKENS)
    if case == "rank":
        x = x[0]
    elif case == "short":
        x = x[:, :1]
    elif case == "dtype":
        x = x.astype(mx.float32)
    elif case == "weights_dtype":
        weights = weights.astype(mx.float32)
    elif case == "shapes":
        weights = weights[..., :6]
    elif case == "prefix":
        indices, weights = indices[:, :1], weights[:, :1]
    elif case == "top_k":
        indices, weights = indices[..., :4], weights[..., :4]
    elif case == "training":
        switch.train()
    elif case == "sharding":
        switch.sharding_group = object()
    assert not patch._should_route(switch, x, indices, weights, patch._MIN_TOKENS)
