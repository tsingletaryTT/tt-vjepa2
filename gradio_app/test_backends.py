# SPDX-License-Identifier: MIT
"""RemoteBackend tests: proves the HTTP client round-trips correctly against the real
FastAPI service app (service/main.py), in-process via starlette.testclient.TestClient
(a sync-compatible ASGI transport -- no real socket, no hardware). TestClient subclasses
httpx.Client, so it's a drop-in for RemoteBackend's real (httpx.Client-based) usage."""

import sys
from pathlib import Path

import numpy as np
import torch
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
from backends import RemoteBackend  # noqa: E402
from main import create_app  # noqa: E402


class _FixedBackend:
    name = "fixed"

    def encode_frame(self, frame_uint8):
        return torch.full((1, 2, 3), 7.0)

    def predict_step(self, reps, actions, states):
        return torch.full((1, 2, 3), 9.0), 12.5


def _make_remote_backend(stub_backend):
    service_app = create_app(stub_backend)
    client = TestClient(service_app)
    return RemoteBackend(client=client)


def test_remote_backend_encode_frame_matches_service_result():
    backend = _make_remote_backend(_FixedBackend())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rep = backend.encode_frame(frame)
    assert rep.shape == (1, 2, 3)
    assert torch.equal(rep, torch.full((1, 2, 3), 7.0))


def test_remote_backend_predict_step_matches_service_result():
    backend = _make_remote_backend(_FixedBackend())
    reps = torch.zeros(1, 1, 2, 3)
    actions = torch.zeros(1, 1, 7)
    states = torch.zeros(1, 1, 7)
    next_rep, latency_ms = backend.predict_step(reps, actions, states)
    assert next_rep.shape == (1, 2, 3)
    assert torch.equal(next_rep, torch.full((1, 2, 3), 9.0))
    assert latency_ms == 12.5
