# SPDX-License-Identifier: Apache-2.0
"""Sliding-window prefill attention: routing is structural, results match."""

import mlx.core as mx
import pytest

from omlx.patches import sliding_window_prefill_attention as sw

WINDOW = 64
BLOCK = 16


def _qkv(q_len, offset, n_q=4, n_kv=2, head_dim=32):
    mx.random.seed(0)
    k_len = offset + q_len
    q = mx.random.normal((1, n_q, q_len, head_dim)).astype(mx.float16)
    k = mx.random.normal((1, n_kv, k_len, head_dim)).astype(mx.float16)
    v = mx.random.normal((1, n_kv, k_len, head_dim)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


def _band_mask(q_len, offset, window=WINDOW):
    """Register a band mask through the wrapped ``create_causal_mask``."""
    from mlx_lm.models.cache import create_causal_mask

    wrapped = sw._wrap_create_causal_mask(create_causal_mask)
    return wrapped(q_len, offset, window_size=window)


@pytest.fixture(autouse=True)
def _clear_registry():
    sw._BANDS.clear()
    yield
    sw._BANDS.clear()


@pytest.mark.parametrize("offset", [0, 40, 200])
def test_windowed_attention_matches_full_mask(offset):
    q_len = 96
    q, k, v = _qkv(q_len, offset)
    mask = _band_mask(q_len, offset)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.2, mask=mask)
    got = sw.windowed_attention(q, k, v, 0.2, mask, offset, WINDOW, BLOCK)
    assert got.shape == expected.shape
    assert mx.allclose(got, expected, atol=2e-3, rtol=2e-3).item()


def test_registered_band_mask_routes():
    q, k, v = _qkv(96, 200)
    mask = _band_mask(96, 200)
    assert sw._should_route(q, k, mask, None, BLOCK)


def test_unregistered_mask_never_routes():
    q, k, v = _qkv(96, 200)
    mask = _band_mask(96, 200)
    # Same values in a different array: e.g. a vision overlay built from it.
    assert not sw._should_route(q, k, mask | mask, None, BLOCK)


def test_padded_masks_are_not_registered():
    from mlx_lm.models.cache import create_causal_mask

    wrapped = sw._wrap_create_causal_mask(create_causal_mask)
    wrapped(96, 0, window_size=WINDOW, left_padding=mx.array([3]))
    wrapped(96, 0, window_size=WINDOW, right_padding=mx.array([3]))
    assert not sw._BANDS


def test_short_chunk_and_causal_string_do_not_route():
    q, k, v = _qkv(BLOCK, 200)
    assert not sw._should_route(q, k, _band_mask(BLOCK, 200), None, BLOCK)
    q, k, v = _qkv(96, 200)
    assert not sw._should_route(q, k, "causal", None, BLOCK)


def test_key_length_must_match_the_registered_band():
    q, k, v = _qkv(96, 200)
    mask = _band_mask(96, 200)
    assert not sw._should_route(q, k[:, :, 1:], mask, None, BLOCK)


def test_sinks_and_quantized_keys_do_not_route():
    q, k, v = _qkv(96, 200)
    mask = _band_mask(96, 200)
    assert not sw._should_route(q, k, mask, mx.zeros((4,)), BLOCK)
    assert not sw._should_route(q, (k, k, k), mask, None, BLOCK)


def test_chunk_inside_the_window_does_not_route():
    q, k, v = _qkv(96, 0, head_dim=32)
    assert not sw._should_route(q, k, _band_mask(96, 0, window=96), None, BLOCK)


@pytest.mark.parametrize(
    "cache_module", ["mlx_lm.models.cache", "mlx_vlm.models.cache"]
)
def test_masks_from_a_real_rotating_cache_route(cache_module, monkeypatch):
    """The producer path: ``RotatingKVCache.make_mask`` -> registered band."""
    import importlib

    module = importlib.import_module(cache_module)
    monkeypatch.setattr(
        module,
        "create_causal_mask",
        sw._wrap_create_causal_mask(module.create_causal_mask),
    )
    cache = module.RotatingKVCache(max_size=WINDOW)
    keys = mx.zeros((1, 2, 96, 32), dtype=mx.float16)
    cache.update_and_fetch(keys, keys)
    # Second chunk: the cache keeps ``window - 1`` old keys next to the new ones.
    mask = cache.make_mask(96, window_size=WINDOW)
    assert isinstance(mask, mx.array)
    q, k, _ = _qkv(96, WINDOW - 1)
    assert mask.shape == (96, k.shape[2])
    assert sw._should_route(q, k, mask, None, BLOCK)


def _fake_base():
    import types

    base = types.SimpleNamespace(calls=[])

    def stock(queries, keys, values, cache, scale, mask, sinks=None):
        base.calls.append((queries.shape[-2], keys.shape[-2]))
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )

    base.scaled_dot_product_attention = stock
    return base, stock


def test_every_block_including_the_tail_goes_through_the_previous_sdpa():
    base, stock = _fake_base()
    sw._patch_sdpa(base, BLOCK)
    patched = base.scaled_dot_product_attention
    assert patched is not stock
    q_len, offset = 100, WINDOW - 1  # 6 full blocks and a 4-row tail
    q, k, v = _qkv(q_len, offset)
    mask = _band_mask(q_len, offset)
    out = patched(q, k, v, None, 0.2, mask)
    assert [rows for rows, _ in base.calls] == [16] * 6 + [4]
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.2, mask=mask)
    assert mx.allclose(out, expected, atol=2e-3, rtol=2e-3).item()


def test_unregistered_masks_reach_the_previous_sdpa_once():
    base, _ = _fake_base()
    sw._patch_sdpa(base, BLOCK)
    q, k, v = _qkv(100, WINDOW - 1)
    mask = _band_mask(100, WINDOW - 1)
    base.scaled_dot_product_attention(q, k, v, None, 0.2, mask | mask)
    assert base.calls == [(100, k.shape[2])]


def test_registry_never_keeps_a_mask_alive():
    mask = _band_mask(96, 200)
    assert len(sw._BANDS) == 1
    del mask
    assert not sw._BANDS


def test_evicted_band_falls_back_to_the_stock_path():
    q, k, v = _qkv(96, 200)
    masks = [_band_mask(96, 200) for _ in range(sw._REGISTRY_SIZE + 1)]
    assert sw._route_band(q, k, masks[0], None, BLOCK) is None
    assert sw._route_band(q, k, masks[-1], None, BLOCK) == (200, WINDOW)


def test_registration_failure_does_not_break_mask_creation(monkeypatch):
    from mlx_lm.models.cache import create_causal_mask

    def boom(*args):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(sw, "_register", boom)
    wrapped = sw._wrap_create_causal_mask(create_causal_mask)
    mask = wrapped(8, 0, window_size=4)
    assert mask.shape == (8, 8)


def test_wrapping_the_mask_builder_twice_is_a_no_op(monkeypatch):
    import types

    from mlx_lm.models.cache import create_causal_mask

    module = types.SimpleNamespace(
        create_causal_mask=sw._wrap_create_causal_mask(create_causal_mask)
    )
    first = module.create_causal_mask
    assert sw._patch_mask(module)
    assert module.create_causal_mask is first


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gemma_like_head_dim_matches_full_mask(dtype):
    q_len, offset = 150, WINDOW - 1
    q, k, v = _qkv(q_len, offset, n_q=8, n_kv=1, head_dim=256)
    q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
    mask = _band_mask(q_len, offset)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
    got = sw.windowed_attention(q, k, v, 1.0, mask, offset, WINDOW, BLOCK)
    assert mx.allclose(got, expected, atol=5e-2, rtol=5e-2).item()
