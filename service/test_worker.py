# SPDX-License-Identifier: MIT
"""Tests for BackendWorker: routes encode_frame/predict_step/plan_step calls onto a
single dedicated thread, so concurrent callers are serialized against the backend --
this is the service-side equivalent of the concurrency_id fix applied to Gradio, now
protecting any client rather than only Gradio's own tabs."""

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import BackendWorker  # noqa: E402


class _StubBackend:
    name = "stub"

    def __init__(self):
        self.calls: list[str] = []

    def encode_frame(self, frame_uint8):
        return torch.zeros(1, 2, 3)

    def predict_step(self, reps, actions, states):
        return torch.ones(1, 2, 3), 5.0


class _ConcurrencyDetectingBackend:
    """Records whether two predict_step calls ever overlap in wall-clock time --
    the real thing a serialization bug would produce."""

    name = "stub"

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent_seen = 0

    def predict_step(self, reps, actions, states):
        with self._lock:
            self._active += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._active)
        time.sleep(0.05)
        with self._lock:
            self._active -= 1
        return torch.ones(1, 2, 3), 5.0


def test_worker_encode_frame_returns_backend_result():
    async def run():
        worker = BackendWorker(_StubBackend())
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        rep = await worker.encode_frame(frame)
        assert torch.equal(rep, torch.zeros(1, 2, 3))
        worker.shutdown()

    asyncio.run(run())


def test_worker_predict_step_returns_backend_result():
    async def run():
        worker = BackendWorker(_StubBackend())
        next_rep, latency_ms = await worker.predict_step(
            torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 7), torch.zeros(1, 1, 7)
        )
        assert torch.equal(next_rep, torch.ones(1, 2, 3))
        assert latency_ms == 5.0
        worker.shutdown()

    asyncio.run(run())


def test_worker_serializes_concurrent_predict_step_calls():
    async def run():
        backend = _ConcurrencyDetectingBackend()
        worker = BackendWorker(backend)
        args = (torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 7), torch.zeros(1, 1, 7))
        await asyncio.gather(
            worker.predict_step(*args),
            worker.predict_step(*args),
            worker.predict_step(*args),
        )
        assert backend.max_concurrent_seen == 1
        worker.shutdown()

    asyncio.run(run())
