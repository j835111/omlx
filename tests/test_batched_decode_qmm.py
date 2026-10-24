# SPDX-License-Identifier: Apache-2.0
"""Skinny-M quantized matmul routing for plain batched decode.

The verify-shape qmm kernels beat stock qmm once a decode step carries
several rows (B concurrent requests), but were armed only around MTP/DFlash
verify forwards. ``batched_decode_qmm(rows)`` arms them for a normal decode
step when ``rows >= BATCHED_DECODE_MIN_ROWS`` (below that stock qmm is as
fast or faster) and always restores the previous routing state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omlx.patches import qwen35_verify_qmm as vq


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    vq.set_verify_qmm_armed(False)
    # Applying the patch mutates nn.QuantizedLinear process-wide; record only.
    calls = []
    monkeypatch.setattr(vq, "apply_verify_qmm_patch", lambda: calls.append(1) or True)
    yield calls
    vq.set_verify_qmm_armed(False)


def test_min_rows_is_six():
    assert vq.BATCHED_DECODE_MIN_ROWS == 6


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 4, 5])
def test_below_threshold_does_not_arm(rows, _clean_state):
    with vq.batched_decode_qmm(rows):
        assert vq._is_armed() is False
    assert _clean_state == []


@pytest.mark.parametrize("rows", [6, 7, 8, 12, 24])
def test_at_or_above_threshold_arms_and_restores(rows, _clean_state):
    with vq.batched_decode_qmm(rows):
        assert vq._is_armed() is True
        assert vq.is_row_exact_armed() is False
    assert vq._is_armed() is False
    assert _clean_state == [1]  # patch applied lazily on first arm


def test_restores_on_exception():
    with pytest.raises(RuntimeError):
        with vq.batched_decode_qmm(8):
            assert vq._is_armed() is True
            raise RuntimeError("boom")
    assert vq._is_armed() is False


def test_restores_previous_armed_state():
    # An MTP verify forward that already armed the kernels must stay armed
    # after a nested context exits (and vice versa).
    vq.set_verify_qmm_armed(True)
    with vq.batched_decode_qmm(8):
        assert vq._is_armed() is True
    assert vq._is_armed() is True


def test_does_not_override_row_exact_mode():
    vq.set_verify_qmm_armed(True, row_exact=True)
    with vq.batched_decode_qmm(8):
        assert vq.is_row_exact_armed() is True
        assert vq._is_armed() is False
    assert vq.is_row_exact_armed() is True


def test_patch_failure_leaves_stock_path(monkeypatch):
    monkeypatch.setattr(vq, "apply_verify_qmm_patch", lambda: False)
    with vq.batched_decode_qmm(8):
        assert vq._is_armed() is False


def test_patch_exception_does_not_break_decode(monkeypatch):
    def boom():
        raise ImportError("no kernels")

    monkeypatch.setattr(vq, "apply_verify_qmm_patch", boom)
    with vq.batched_decode_qmm(8):
        assert vq._is_armed() is False


def test_arming_is_thread_local():
    import threading

    seen = {}

    def worker():
        seen["other"] = vq._is_armed()

    with vq.batched_decode_qmm(8):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert vq._is_armed() is True
    assert seen["other"] is False


# --- scheduler hook --------------------------------------------------------


@pytest.fixture
def scheduler_step(monkeypatch):
    from omlx import scheduler

    seen = {}

    def fake_original(self):
        seen["armed_during_step"] = vq._is_armed()
        return "step-result"

    monkeypatch.setattr(scheduler, "_original_generation_batch_step", fake_original)
    monkeypatch.setattr(scheduler, "_omlx_realign_generation_batch_rows", lambda s: None)
    monkeypatch.setattr(scheduler, "_omlx_advance_grammar_rows", lambda s: None)
    return scheduler, seen


def _batch(n):
    return SimpleNamespace(model=SimpleNamespace(), uids=list(range(n)))


@pytest.mark.parametrize("n,expected", [(1, False), (5, False), (6, True), (8, True)])
def test_generation_step_arms_by_batch_size(scheduler_step, n, expected):
    scheduler, seen = scheduler_step
    assert scheduler._patched_generation_batch_step(_batch(n)) == "step-result"
    assert seen["armed_during_step"] is expected
    assert vq._is_armed() is False


def test_generation_step_disarms_when_step_raises(scheduler_step, monkeypatch):
    scheduler, _ = scheduler_step

    def fail(self):
        assert vq._is_armed() is True
        raise ValueError("step failed")

    monkeypatch.setattr(scheduler, "_original_generation_batch_step", fail)
    with pytest.raises(ValueError):
        scheduler._patched_generation_batch_step(_batch(8))
    assert vq._is_armed() is False
