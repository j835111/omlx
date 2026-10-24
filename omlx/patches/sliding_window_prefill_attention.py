# SPDX-License-Identifier: Apache-2.0
"""Skip out-of-window keys in sliding-window prefill attention.

A model with sliding-window layers (Gemma, Mistral, GPT-OSS, ...) hands every
prefill chunk longer than its window an explicit boolean band mask, and MLX's
fused SDPA then walks every key block of the chunk and masks most of them to
nothing. Cost grows with ``L * S`` instead of ``L * window``: a 2048-token
chunk on a 1024-token window runs at about 1 TFLOPS against 2.5 for plain
causal attention.

This splits the queries into blocks and gives each block only the keys its
band can reach (``window + block - 1`` of them) together with the matching
slice of the same mask. Keys outside the band carry zero weight under the
original mask, so each block computes the original attention; results differ
from the full call only in reduction order (bf16 rounding level).

Only masks that ``create_causal_mask`` built from ``(N, offset, window_size)``
alone are routed. Those are registered when created; any other mask (batch
padding, Gemma's bidirectional vision overlay, an additive float mask) is a
different array, is not registered, and takes the stock path, so a mask that
is not a plain band can never be cut. It is pure MLX and needs no native
kernel, so it applies to every Apple Silicon GPU.

Each block goes through the SDPA that was installed before this patch, so
workarounds in that chain (the M1/M2 vector-kernel split in
:mod:`omlx.patches.sdpa256_attention`, quantized-cache handling) still apply
to the block-sized calls. A registered mask is assumed not to be modified in
place afterwards; no producer in mlx-lm or mlx-vlm does that.
"""

from __future__ import annotations

import inspect
import logging
import sys
import weakref
from collections import OrderedDict
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
# Query rows per block. 128-256 measured best from 2k to 4k-token chunks on
# M4 (64 costs more dispatches, 512 keeps too many dead keys).
_DEFAULT_BLOCK = 256
# A forward builds one band mask per window size. Eviction just falls back to
# the stock path.
_REGISTRY_SIZE = 8
_REBIND_PREFIXES = ("mlx_lm.", "mlx_vlm.", "omlx.")
# id(mask) -> (weakref to mask, offset, window). The registry never keeps a
# mask alive; an entry disappears with its mask, so a reused id cannot match.
_BANDS: OrderedDict[int, tuple[weakref.ref, int, int]] = OrderedDict()


def _register(mask: mx.array, offset: int, window: int) -> None:
    key = id(mask)
    _BANDS[key] = (
        weakref.ref(mask, lambda _, key=key: _BANDS.pop(key, None)),
        offset,
        window,
    )
    while len(_BANDS) > _REGISTRY_SIZE:
        _BANDS.popitem(last=False)


def _wrap_create_causal_mask(original):
    signature = inspect.signature(original)

    def create_causal_mask(*args, **kwargs):
        mask = original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments
            rows = params["N"]
            offset = params.get("offset", 0)
            window = params.get("window_size")
            padded = (
                params.get("right_padding") is not None
                or params.get("left_padding") is not None
            )
        except (TypeError, KeyError):
            return mask
        if (
            window
            and rows > 1
            and not padded
            and isinstance(mask, mx.array)
            and mask.ndim == 2
        ):
            try:
                _register(mask, int(offset), int(window))
            except Exception:
                logger.debug("band mask not registered", exc_info=True)
        return mask

    create_causal_mask._omlx_band_mask = True
    return create_causal_mask


def _band_of(mask) -> tuple[int, int] | None:
    entry = _BANDS.get(id(mask))
    if entry is None or entry[0]() is not mask:
        return None
    return entry[1], entry[2]


def _route_band(queries, keys, mask, sinks, block: int) -> tuple[int, int] | None:
    """Return the registered ``(offset, window)`` when this call can be cut."""
    if sinks is not None or queries.ndim != 4:
        return None
    if not isinstance(keys, mx.array) or not isinstance(mask, mx.array):
        return None
    q_len, k_len = queries.shape[-2], keys.shape[-2]
    if q_len <= block or mask.shape != (q_len, k_len):
        return None
    band = _band_of(mask)
    if band is None:
        return None
    offset, window = band
    # The registered band must describe exactly these keys: ``k_len`` keys
    # ending at the last query row, and some key must be out of reach.
    if offset + q_len == k_len and k_len > window + block:
        return band
    return None


def _should_route(queries, keys, mask, sinks, block: int) -> bool:
    return _route_band(queries, keys, mask, sinks, block) is not None


def windowed_attention(
    queries, keys, values, scale, mask, offset, window, block, attend=None
):
    """Block-wise SDPA over a ``window`` band; ``mask`` is the full band mask.

    ``attend(q, k, v, mask)`` computes one block; it defaults to MLX's fused
    SDPA and is how the caller keeps the rest of its SDPA chain in the loop.
    """
    if attend is None:

        def attend(q, k, v, m):
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)

    q_len = queries.shape[-2]
    outputs = []
    for start in range(0, q_len, block):
        stop = min(start + block, q_len)
        first = max(0, offset + start - window + 1)
        last = offset + stop
        outputs.append(
            attend(
                queries[:, :, start:stop],
                keys[:, :, first:last],
                values[:, :, first:last],
                mask[start:stop, first:last],
            )
        )
    return mx.concatenate(outputs, axis=2)


def _rebind(old, new) -> None:
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(_REBIND_PREFIXES):
            continue
        for name in ("scaled_dot_product_attention", "create_causal_mask"):
            try:
                if getattr(module, name, None) is old:
                    setattr(module, name, new)
            except Exception:
                continue


def _patch_sdpa(base: Any, block: int) -> bool:
    original = getattr(base, "scaled_dot_product_attention", None)
    if original is None:
        return False

    def scaled_dot_product_attention(
        queries, keys, values, cache, scale: float, mask, sinks=None
    ):
        band = _route_band(queries, keys, mask, sinks, block)
        if band is not None:
            return windowed_attention(
                queries,
                keys,
                values,
                scale,
                mask,
                *band,
                block,
                attend=lambda q, k, v, m: original(q, k, v, cache, scale, m, None),
            )
        return original(queries, keys, values, cache, scale, mask, sinks)

    scaled_dot_product_attention._omlx_sliding_window = True
    base.scaled_dot_product_attention = scaled_dot_product_attention
    _rebind(original, scaled_dot_product_attention)
    return True


def _patch_mask(module: Any) -> bool:
    original = getattr(module, "create_causal_mask", None)
    if original is None or getattr(original, "_omlx_band_mask", False):
        return original is not None
    wrapped = _wrap_create_causal_mask(original)
    module.create_causal_mask = wrapped
    _rebind(original, wrapped)
    return True


def apply_sliding_window_prefill_attention_patch(block: int | None = None) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    block = int(block or _DEFAULT_BLOCK)
    patched = False
    try:
        from mlx_lm.models import base as lm_base
        from mlx_lm.models import cache as lm_cache

        patched |= _patch_mask(lm_cache)
        patched |= _patch_sdpa(lm_base, block)
    except ImportError:
        pass
    try:
        from mlx_vlm.models import base as vlm_base
        from mlx_vlm.models import cache as vlm_cache

        patched |= _patch_mask(vlm_cache)
        patched |= _patch_sdpa(vlm_base, block)
    except ImportError:
        pass
    if patched:
        _PATCHED = True
        logger.info("Sliding-window prefill attention patch applied (block=%d)", block)
    return patched


__all__ = ["apply_sliding_window_prefill_attention_patch", "windowed_attention"]
