# SPDX-License-Identifier: MIT
"""Endpoint tests for the FastAPI service, via httpx.ASGITransport (in-process, no
real socket, no hardware) against a stub backend. Covers /encode, /predict_step,
/plan_step round-trip correctness and the error-handling contract."""

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradio_app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import create_app  # noqa: E402
from test_planning import _StubBackend  # noqa: E402
from wire import frame_to_b64, tensor_from_b64, tensor_to_b64  # noqa: E402


class _FixedBackend:
    name = "fixed"

    def encode_frame(self, frame_uint8):
        return torch.full((1, 2, 3), 7.0)

    def predict_step(self, reps, actions, states):
        return torch.full((1, 2, 3), 9.0), 12.5


class _RaisingBackend:
    name = "raising"

    def predict_step(self, reps, actions, states):
        raise RuntimeError("boom: shape mismatch")


async def _post(app, path, json):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=json)


def test_encode_endpoint_returns_backend_result():
    app = create_app(_FixedBackend())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    resp = asyncio.run(_post(app, "/encode", {"frame_png_b64": frame_to_b64(frame)}))
    assert resp.status_code == 200
    rep = tensor_from_b64(resp.json()["rep"])
    assert torch.equal(rep, torch.full((2, 3), 7.0))


def test_predict_step_endpoint_returns_backend_result():
    app = create_app(_FixedBackend())
    body = {
        "reps": tensor_to_b64(torch.zeros(1, 1, 2, 3)),  # [B,T,HW,D], B=1,T=1
        "actions": tensor_to_b64(torch.zeros(1, 1, 7)),
        "states": tensor_to_b64(torch.zeros(1, 1, 7)),
    }
    resp = asyncio.run(_post(app, "/predict_step", body))
    assert resp.status_code == 200
    payload = resp.json()
    assert torch.equal(tensor_from_b64(payload["next_rep"]), torch.full((1, 2, 3), 9.0))
    assert payload["latency_ms"] == 12.5


def test_predict_step_endpoint_preserves_batch_dimension():
    """reps/actions/states are NOT stripped to a batch-of-1 convention -- they travel
    as-is, because planning.cem_search calls predict_step with the CEM sample count as
    the batch dimension. An earlier version of this endpoint assumed batch-of-1 here,
    which produced a real 422 once cem_search's batched calls actually hit it (see
    gradio_app/test_backends.py::test_remote_backend_predict_step_handles_batched_samples)."""
    hw, d, samples = 2, 3, 4
    app = create_app(_StubBackend(hw, d))
    reps = torch.arange(samples * hw * d, dtype=torch.float32).reshape(samples, 1, hw, d)
    actions = torch.arange(samples * 7, dtype=torch.float32).reshape(samples, 1, 7)
    states = torch.zeros(samples, 1, 7)
    body = {"reps": tensor_to_b64(reps), "actions": tensor_to_b64(actions), "states": tensor_to_b64(states)}

    resp = asyncio.run(_post(app, "/predict_step", body))

    assert resp.status_code == 200
    next_rep = tensor_from_b64(resp.json()["next_rep"])
    assert next_rep.shape == (samples, hw, d)
    expected = actions.mean(dim=(1, 2)).view(samples, 1, 1).expand(samples, hw, d)
    assert torch.allclose(next_rep, expected)


def test_plan_step_endpoint_returns_sane_response():
    hw, d = 2, 3
    app = create_app(_StubBackend(hw, d))
    reps = torch.zeros(1, hw, d)  # [T, HW, D], T=1
    body = {
        "reps": tensor_to_b64(reps),
        "prior_actions": [],
        "states_seq": [[0.0] * 7],
        "cur_pose": [0.0] * 7,
        "target_action": [0.05, -0.02, 0.03, 0, 0, 0, 0.1],
        "cem_steps": 2,
        "samples": 4,
        "topk": 2,
        "maxnorm": 0.1,
    }
    resp = asyncio.run(_post(app, "/plan_step", body))
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["found_action"]) == 7
    assert isinstance(payload["energy"], float)
    assert isinstance(payload["latency_ms"], float)
    tensor_from_b64(payload["next_rep"])  # decodes without error


def test_backend_exception_returns_500_with_error_message():
    app = create_app(_RaisingBackend())
    body = {
        "reps": tensor_to_b64(torch.zeros(1, 1, 2, 3)),
        "actions": tensor_to_b64(torch.zeros(1, 1, 7)),
        "states": tensor_to_b64(torch.zeros(1, 1, 7)),
    }
    resp = asyncio.run(_post(app, "/predict_step", body))
    assert resp.status_code == 500
    assert "boom: shape mismatch" in resp.json()["error"]
