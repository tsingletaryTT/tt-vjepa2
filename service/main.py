# SPDX-License-Identifier: MIT
"""FastAPI app exposing the encoder/predictor/planner as a standalone ASGI service.
Owns one backend instance (Reference or TTNN) via a BackendWorker, which serializes
every device-touching call onto a single dedicated thread -- see worker.py's docstring
for why. Endpoints mirror the in-process primitives (encode_frame, predict_step) plus
the higher-level plan_step, per docs/superpowers/specs/2026-09-08-asgi-service-design.md.

Wire convention: /encode is always a single frame, so its response omits the
leading batch-of-1 dimension the in-process call uses (rep is [HW,D] on the wire, not
[1,HW,D]) -- this module adds it back before calling the backend and strips it again
before responding. /predict_step's reps/actions/states/next_rep are NOT batch-stripped
-- they travel as safetensors preserving whatever batch dimension the caller actually
used, unchanged, because that batch dimension isn't always 1: planning.cem_search
calls predict_step with the CEM sample count as the batch dim (see
gradio_app/test_backends.py::test_remote_backend_predict_step_handles_batched_samples,
added after a real 422 surfaced this -- an earlier version of this endpoint assumed
batch-of-1 here too, which silently broke the CEM Planning tab and the Dance tab's
"Plan with CEM" toggle under --backend remote).
"""

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wire import frame_from_b64, tensor_from_b64, tensor_to_b64  # noqa: E402
from worker import BackendWorker  # noqa: E402


class EncodeRequest(BaseModel):
    frame_png_b64: str


class EncodeResponse(BaseModel):
    rep: str


class PredictStepRequest(BaseModel):
    reps: str
    actions: str
    states: str


class PredictStepResponse(BaseModel):
    next_rep: str
    latency_ms: float


class PlanStepRequest(BaseModel):
    reps: str
    prior_actions: list[list[float]]
    states_seq: list[list[float]]
    cur_pose: list[float]
    target_action: list[float]
    cem_steps: int = 6
    samples: int = 12
    topk: int = 4
    maxnorm: float = 0.12


class PlanStepResponse(BaseModel):
    found_action: list[float]
    next_rep: str
    energy: float
    latency_ms: float


def create_app(backend) -> FastAPI:
    """backend: anything implementing the ReferenceBackend/TTNNBackend duck type
    (.name, .encode_frame(), .predict_step())."""
    app = FastAPI()
    worker = BackendWorker(backend)
    app.state.worker = worker

    # Caught here, per-endpoint, rather than via a Starlette exception-handler
    # middleware: ServerErrorMiddleware re-raises after sending the response (so a
    # real ASGI server can log it), which is fine for uvicorn but means a bare
    # httpx.ASGITransport test (no server process in between) sees the exception
    # instead of the response. A plain try/except returns the same {"error": ...}
    # body identically in both cases.
    @app.post("/encode", response_model=EncodeResponse)
    async def encode(req: EncodeRequest):
        try:
            frame = frame_from_b64(req.frame_png_b64)
            rep = await worker.encode_frame(frame)
            return EncodeResponse(rep=tensor_to_b64(rep.squeeze(0)))
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/predict_step", response_model=PredictStepResponse)
    async def predict_step(req: PredictStepRequest):
        try:
            reps = tensor_from_b64(req.reps)
            actions = tensor_from_b64(req.actions)
            states = tensor_from_b64(req.states)
            next_rep, latency_ms = await worker.predict_step(reps, actions, states)
            return PredictStepResponse(next_rep=tensor_to_b64(next_rep), latency_ms=latency_ms)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/plan_step", response_model=PlanStepResponse)
    async def plan_step_endpoint(req: PlanStepRequest):
        try:
            reps = tensor_from_b64(req.reps).unsqueeze(0)
            prior_actions = [np.array(a, dtype=np.float32) for a in req.prior_actions]
            states_seq = [np.array(s, dtype=np.float32) for s in req.states_seq]
            cur_pose = np.array(req.cur_pose, dtype=np.float32)
            target_action = np.array(req.target_action, dtype=np.float32)
            found_action, next_rep, energy, latency_ms = await worker.plan_step(
                reps, prior_actions, states_seq, cur_pose, target_action,
                cem_steps=req.cem_steps, samples=req.samples, topk=req.topk, maxnorm=req.maxnorm,
            )
            return PlanStepResponse(
                found_action=found_action.tolist(),
                next_rep=tensor_to_b64(next_rep.squeeze(0)),
                energy=energy,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ttnn", "reference"], default="ttnn")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    gradio_app_dir = Path(__file__).resolve().parent.parent / "gradio_app"
    sys.path.insert(0, str(gradio_app_dir))
    if args.backend == "ttnn":
        from backends import TTNNBackend

        backend = TTNNBackend(device_id=args.device_id)
    else:
        from backends import ReferenceBackend

        backend = ReferenceBackend()

    app = create_app(backend)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
