# SPDX-License-Identifier: Apache-2.0
"""Route structurally eligible affine q4 linears during long prefill."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .qwen35_q4_mlp import (
    _MIN_TOKENS,
    _QMM_VARIANT,
    _can_route_affine_linear,
    _has_native_qmm,
    _is_supported_affine_linear_shape,
    _linear_qmm,
)

logger = logging.getLogger(__name__)

_Q8_MIN_TOKENS = 2**63 - 1
# Families with their own q4 prefill routing and decode fast paths that test
# `type(linear) is nn.QuantizedLinear`; swapping the class would disable them.
_DEDICATED_FAMILIES = (
    "mlx_vlm.models.qwen",
    "mlx_lm.models.qwen",
    "qwen4_exp",
)
_ROUTED_CLASSES: dict[type, type] = {}
_PROBE_CACHE: dict[tuple[int, int, int, int, Any], bool] = {}


def _probe(module: nn.QuantizedLinear, input_dim: int) -> bool:
    """Compare stock and native calls before the instance is rewritten."""
    try:
        # Always check BF16; also exercise the activation dtype matching the
        # scales so FP16 weights cannot pass via _linear_qmm's stock fallback.
        dtypes = {mx.bfloat16, module.scales.dtype}
        for dtype in dtypes:
            x = mx.random.normal((1, 64, input_dim), key=mx.random.key(0)).astype(dtype)
            original = nn.QuantizedLinear.__call__(module, x)
            native = _linear_qmm(module, x, _QMM_VARIANT)
            mx.eval(original, native)
            if original.shape != native.shape or original.dtype != native.dtype:
                logger.debug("Quantized linear prefill probe shape/dtype mismatch")
                return False
            original, native = original.astype(mx.float32), native.astype(mx.float32)
            error = mx.max(mx.abs(native - original))
            tolerance = 1e-2 * mx.max(mx.abs(original)) + 1e-3
            if not (error <= tolerance).item():
                logger.debug("Quantized linear prefill probe numerical mismatch")
                return False
        return True
    except Exception:
        logger.debug("Quantized linear prefill probe failed", exc_info=True)
        return False


def _routed_class(original: type) -> type:
    if original in _ROUTED_CLASSES:
        return _ROUTED_CLASSES[original]

    def routed_call(self, x):
        # Decode exits before inspecting quantization state or native kernels.
        if x.ndim != 3 or x.shape[-2] < _MIN_TOKENS:
            return super(routed, self).__call__(x)
        if not self._omlx_q4_prefill_failed and _can_route_affine_linear(
            self, x, _MIN_TOKENS, _Q8_MIN_TOKENS
        ):
            try:
                return _linear_qmm(self, x, _QMM_VARIANT)
            except Exception:
                self._omlx_q4_prefill_failed = True
                logger.warning(
                    "Quantized linear prefill fast path failed", exc_info=True
                )
        return super(routed, self).__call__(x)

    # Like the other structural patches, model instances are never pickled.
    routed = type(
        original.__name__,
        (original,),
        {
            "__module__": original.__module__,
            "__qualname__": original.__qualname__,
            "__call__": routed_call,
            "_omlx_q4_prefill": True,
            "_omlx_q4_prefill_failed": False,
        },
    )
    _ROUTED_CLASSES[original] = routed
    return routed


def apply_quantized_linear_prefill(model: Any) -> int:
    """Rewrite eligible stock q4 linears in place; return their count."""
    if not mx.metal.is_available() or not _has_native_qmm():
        return 0
    modules = list(model.named_modules())
    if any(
        any(tag in type(module).__module__ for tag in _DEDICATED_FAMILIES)
        for _, module in modules
    ):
        logger.debug("Quantized linear prefill skipped: dedicated q4 routing")
        return 0
    count = 0
    for name, module in modules:
        if name.endswith(("lm_head", "embed_tokens", "embed_tokens_per_layer")):
            continue
        if type(module) is not nn.QuantizedLinear or module.bits != 4:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != mx.uint32 or weight.ndim != 2:
            continue
        input_dim = weight.shape[1] * 8
        scales = getattr(module, "scales", None)
        if scales is None or not _is_supported_affine_linear_shape(
            module, scales.dtype, 3, 64, input_dim
        ):
            continue
        key = (input_dim, weight.shape[0], module.bits, module.group_size, scales.dtype)
        if key not in _PROBE_CACHE:
            _PROBE_CACHE[key] = _probe(module, input_dim)
        if not _PROBE_CACHE[key]:
            continue
        module.__class__ = _routed_class(type(module))
        count += 1
    if count:
        logger.info(
            "Quantized linear prefill applied: %d modules (min_tokens=%d)",
            count,
            _MIN_TOKENS,
        )
    return count


__all__ = ["apply_quantized_linear_prefill"]
