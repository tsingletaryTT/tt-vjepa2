# SPDX-License-Identifier: MIT
"""FastAPI app exposing the encoder/predictor/planner as a standalone ASGI service, for
tt-model-manager's ``tt-dit-server`` kind (no tokens/KV-cache/continuous batching -- an
action-conditioned world model has nothing for vLLM to do). Owns one backend instance
(Reference or TTNN) via a BackendWorker, which serializes every device-touching call
onto a single dedicated thread -- see worker.py's docstring for why. Endpoints mirror
the in-process primitives (encode_frame, predict_step) plus the higher-level plan_step,
per docs/superpowers/specs/2026-09-08-asgi-service-design.md.

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

TWO CONTRACTS THE MODULE-LEVEL ``app`` HAS TO HONOUR, same as every other tt-dit-server
kind in this org (tt-animatediff/tt-skyreels/tt-waypoint):
1. **Readiness is the lifespan.** tt-model-manager decides the server is up when uvicorn
   logs "Application startup complete", printed only after ASGI lifespan startup
   returns -- so the device open and weight load happen in ``_lifespan`` below, not at
   import time.
2. **Importing this module must not touch hardware.** A manifest's ``verify`` step
   imports the ``app`` attribute at bundle-build time, on a machine with no card. Every
   ttnn / gradio_app.backends import stays inside a function, never at module scope.

``create_app(backend)`` is the other entry point: an eager factory around an
already-constructed backend, for tests (stub backends, no device) and any caller that
already holds one. It shares route registration with the module-level ``app`` via
``_register_routes`` so the two can't drift apart.
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from service.wire import frame_from_b64, tensor_from_b64, tensor_to_b64
from service.worker import BackendWorker


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


def _register_routes(app: FastAPI) -> None:
    """Registers /encode, /predict_step, /plan_step on `app`, reading the worker from
    `app.state.worker` at REQUEST time rather than closing over it as a parameter --
    that's what lets this be called at module-definition time, before the module-level
    `app` below has a worker at all (it gets one once `_lifespan` runs). Shared by
    `create_app` (which sets `app.state.worker` immediately, before any request can
    arrive) and the module-level `app` (which sets it lazily in `_lifespan`), so the two
    route sets can't drift apart.

    Errors are caught here, per-endpoint, rather than via a Starlette exception-handler
    middleware: ServerErrorMiddleware re-raises after sending the response (so a
    real ASGI server can log it), which is fine for uvicorn but means a bare
    httpx.ASGITransport test (no server process in between) sees the exception
    instead of the response. A plain try/except returns the same {"error": ...}
    body identically in both cases."""

    @app.post("/encode", response_model=EncodeResponse)
    async def encode(req: EncodeRequest):
        try:
            frame = frame_from_b64(req.frame_png_b64)
            rep = await app.state.worker.encode_frame(frame)
            return EncodeResponse(rep=tensor_to_b64(rep.squeeze(0)))
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/predict_step", response_model=PredictStepResponse)
    async def predict_step(req: PredictStepRequest):
        try:
            reps = tensor_from_b64(req.reps)
            actions = tensor_from_b64(req.actions)
            states = tensor_from_b64(req.states)
            next_rep, latency_ms = await app.state.worker.predict_step(reps, actions, states)
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
            found_action, next_rep, energy, latency_ms = await app.state.worker.plan_step(
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


def create_app(backend) -> FastAPI:
    """Eager variant: backend: anything implementing the ReferenceBackend/TTNNBackend
    duck type (.name, .encode_frame(), .predict_step()). Used by tests (stub backends,
    no device) and any caller that already holds a constructed backend."""
    app = FastAPI()
    app.state.worker = BackendWorker(backend)
    _register_routes(app)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Real-deployment path: constructs the backend (and opens the device, for
    ``ttnn``) at ASGI startup, not at import time -- see the module docstring's two
    contracts. Backend choice/device id come from env, matching how run.sh (generated
    by tt-model package-thin) sets serving env rather than passing CLI args."""
    backend_kind = os.environ.get("VJEPA2_BACKEND", "ttnn")
    device_id = int(os.environ.get("VJEPA2_DEVICE_ID", "0"))
    if backend_kind == "ttnn":
        from gradio_app.backends import TTNNBackend

        backend = TTNNBackend(device_id=device_id)
    else:
        from gradio_app.backends import ReferenceBackend

        backend = ReferenceBackend()

    app.state.worker = BackendWorker(backend)
    try:
        yield
    finally:
        app.state.worker.shutdown()
        if hasattr(backend, "close"):
            backend.close()


#: The ASGI target `tt-model package-thin --app service.main:app` serves with uvicorn.
#: Importable with zero hardware touched (routes read `app.state.worker`, set only once
#: `_lifespan` has actually run) -- see the module docstring's contract 2.
app = FastAPI(lifespan=_lifespan)
_register_routes(app)


def main():
    """Local dev entry point: `python -m service.main --backend ttnn --device-id 0`.
    Builds its own eager app via create_app -- the packaged/tt-model-manager path uses
    the module-level `app` + `_lifespan` above instead, not this function."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ttnn", "reference"], default="ttnn")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.backend == "ttnn":
        from gradio_app.backends import TTNNBackend

        backend = TTNNBackend(device_id=args.device_id)
    else:
        from gradio_app.backends import ReferenceBackend

        backend = ReferenceBackend()

    dev_app = create_app(backend)
    try:
        uvicorn.run(dev_app, host=args.host, port=args.port)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
