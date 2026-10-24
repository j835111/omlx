# SPDX-License-Identifier: Apache-2.0
"""Combine sorted MoE expert outputs directly with the native weighted sum.

The structural gate matches wrappers with exactly one key, a
stock mlx-lm or mlx-vlm ``SwitchGLU``, and a three-argument forward. Qwen's
``SparseMoeBlock`` owns a router and shared expert as well, so it deliberately
does not match and keeps its own weighted-sum patch.

Results differ from stock by accumulation order and rounding: native accumulates
once in fp32, while stock accumulates in the weights dtype. Decode and short
prefill chunks use the original call.
"""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

import mlx.core as mx

from .qwen35_moe_weighted_sum import (
    _native_switch_weighted_sum,
    _native_weighted_sum,
)

logger = logging.getLogger(__name__)

# Gemma-4 26B-A4B layer on M4: routed is 4-5% faster from 256 tokens up (12% at
# 128) and the prefill memory guard shrinks chunks below 1024 under pressure.
_MIN_TOKENS = 256
_ROUTED_CLASSES: dict[type, type] = {}
_PROBE_CACHE: dict[tuple[type, type, Any], bool] = {}


def _switch_glu_types() -> tuple[type, ...]:
    classes = []
    try:
        from mlx_lm.models.switch_layers import SwitchGLU

        classes.append(SwitchGLU)
    except ImportError:
        pass
    try:
        from mlx_vlm.models.switch_layers import SwitchGLU

        classes.append(SwitchGLU)
    except ImportError:
        pass
    return tuple(classes)


def _switch_child(module: Any, switch_types: tuple[type, ...]) -> Any:
    if list(module.keys()) != ["switch_glu"]:
        return None
    child = module["switch_glu"]
    if isinstance(child, switch_types) and any(
        type(child).__call__ is stock.__call__ for stock in switch_types
    ):
        return child
    return None


def _matches_structure(module: Any, switch_types: tuple[type, ...]) -> bool:
    if not set(vars(module)) <= {"_no_grad", "_training"}:
        return False
    if _switch_child(module, switch_types) is None:
        return False
    try:
        parameters = list(inspect.signature(type(module).__call__).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(parameters) == 4 and all(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
    )


def _compute_dtype(switch_glu: Any) -> Any:
    """Activation dtype the expert projections run in (None if unsupported)."""
    for name in ("gate_up_proj", "gate_proj", "up_proj"):
        projection = getattr(switch_glu, name, None)
        if projection is None:
            continue
        array = projection["scales"] if "scales" in projection else projection["weight"]
        return array.dtype if array.dtype in (mx.float16, mx.bfloat16) else None
    return None


def _probe(module: Any, weighted_sum: Any, dtype: Any = None) -> bool:
    """Compare the original body and native path without changing model state."""
    try:
        if not mx.metal.is_available():
            logger.debug("MoE expert combine probe requires Metal")
            return False
        candidate = copy.deepcopy(module)
        switch_glu = candidate.switch_glu
        dtype = dtype or _compute_dtype(switch_glu)
        if dtype is None:
            logger.debug("MoE expert combine probe has no supported dtype")
            return False
        gate = getattr(switch_glu, "gate_proj", None)
        dims = getattr(gate, "input_dims", None)
        if dims is None:
            dims = getattr(getattr(switch_glu, "up_proj", None), "input_dims", None)
        if dims is None:
            logger.debug("MoE expert combine probe has no input dimensions")
            return False
        experts = gate.num_experts
        x = mx.random.normal((1, 64, dims), key=mx.random.key(0)).astype(dtype)
        logits = mx.random.normal((1, 64, experts), key=mx.random.key(1))
        indices = mx.argpartition(logits, kth=-8, axis=-1)[..., -8:]
        weights = mx.softmax(
            mx.take_along_axis(logits, indices, axis=-1), axis=-1
        ).astype(dtype)
        original = type(module).__call__(candidate, x, indices, weights)
        native = _native_switch_weighted_sum(
            switch_glu, x, indices, weights, weighted_sum
        )
        mx.eval(original, native)
        if original.shape != native.shape or original.dtype != native.dtype:
            logger.debug("MoE expert combine probe shape/dtype mismatch")
            return False
        original, native = original.astype(mx.float32), native.astype(mx.float32)
        error = mx.max(mx.abs(native - original))
        tolerance = 0.05 * mx.max(mx.abs(original)) + 1e-3
        if not (error <= tolerance).item():
            logger.debug("MoE expert combine probe numerical mismatch")
            return False
        return True
    except Exception:
        logger.debug("MoE expert combine probe failed", exc_info=True)
        return False


def _should_route(
    switch_glu: Any,
    x: mx.array,
    indices: mx.array,
    weights: mx.array,
    min_tokens: int,
) -> bool:
    # Decode exits before inspecting expert state or touching Metal.
    if x.ndim != 3 or x.shape[-2] < min_tokens:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if weights.dtype != x.dtype:
        return False
    if indices.shape != weights.shape or indices.shape[:-1] != x.shape[:-1]:
        return False
    top_k = indices.shape[-1]
    if top_k not in (6, 8, 10) or x.shape[-2] * top_k < 64:
        return False
    if switch_glu is None or switch_glu.training:
        return False
    if getattr(switch_glu, "sharding_group", None) is not None:
        return False
    if not mx.metal.is_available() or not hasattr(switch_glu, "down_proj"):
        return False
    return hasattr(switch_glu, "gate_up_proj") or (
        hasattr(switch_glu, "up_proj") and hasattr(switch_glu, "gate_proj")
    )


def _routed_class(original: type, switch_types: tuple[type, ...]) -> type:
    if original in _ROUTED_CLASSES:
        return _ROUTED_CLASSES[original]

    def routed_call(self, x, indices, weights):
        if getattr(self, "_omlx_expert_combine_failed", False):
            return super(routed, self).__call__(x, indices, weights)
        # Keep the common decode path free of native kernel lookups.
        if x.ndim != 3 or x.shape[-2] < _MIN_TOKENS:
            return super(routed, self).__call__(x, indices, weights)
        switch_glu = _switch_child(self, switch_types)
        weighted_sum = _native_weighted_sum()
        if weighted_sum is None or not _should_route(
            switch_glu, x, indices, weights, _MIN_TOKENS
        ):
            return super(routed, self).__call__(x, indices, weights)
        try:
            return _native_switch_weighted_sum(
                switch_glu, x, indices, weights, weighted_sum
            )
        except Exception:
            self._omlx_expert_combine_failed = True
            logger.warning("MoE expert combine fast path failed", exc_info=True)
            return super(routed, self).__call__(x, indices, weights)

    # The dynamic class shadows the original for pickle; oMLX does not pickle models.
    routed = type(
        original.__name__,
        (original,),
        {
            "__module__": original.__module__,
            "__qualname__": original.__qualname__,
            "__call__": routed_call,
            "_omlx_expert_combine": True,
            # Class default keeps the decode hot path off nn.Module.__getattr__.
            "_omlx_expert_combine_failed": False,
        },
    )
    _ROUTED_CLASSES[original] = routed
    return routed


def apply_moe_expert_combine(model: Any) -> int:
    """Rewrite eligible expert-combine wrappers in place; return their count."""
    weighted_sum = _native_weighted_sum()
    if weighted_sum is None:
        return 0
    switch_types = _switch_glu_types()
    count = 0
    for _, module in model.named_modules():
        original = type(module)
        if getattr(original, "_omlx_expert_combine", False):
            continue
        if not _matches_structure(module, switch_types):
            continue
        switch_glu = module.switch_glu
        dtype = _compute_dtype(switch_glu)
        if dtype is None:
            continue
        pair = (original, type(switch_glu), dtype)
        if pair not in _PROBE_CACHE:
            _PROBE_CACHE[pair] = _probe(module, weighted_sum, dtype)
        if not _PROBE_CACHE[pair]:
            continue
        module.__class__ = _routed_class(original, switch_types)
        count += 1
    if count:
        logger.info(
            "MoE expert combine applied: %d modules (min_tokens=%d)", count, _MIN_TOKENS
        )
    return count


__all__ = ["apply_moe_expert_combine"]
