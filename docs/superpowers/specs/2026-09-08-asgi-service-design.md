# ASGI service for the V-JEPA2-AC encoder/predictor/planner

Status: approved for implementation
Date: 2026-09-08

## Problem

Today, `TTNNBackend`/`ReferenceBackend` (`gradio_app/backends.py`) are instantiated
directly inside the Gradio app, and every tab's callback closes over that one shared
instance. Nothing outside the Gradio process can use the encoder, predictor, or CEM
planner, and the device itself is only ever reachable through Gradio's own queue
machinery (which we just had to patch with `concurrency_id`/`concurrency_limit=1` to
stop different tabs from racing each other on the same `ttnn.Device`).

We want other things — future descendant-model work, other consumers, scripts — to be
able to use this model as a building block without going through Gradio at all, and we
want device-access serialization enforced centrally (protecting *any* client) rather
than as a Gradio-specific workaround.

## Goals

- A standalone ASGI service that owns one backend instance (Reference or TTNN, chosen
  at startup) and holds the device/gozer lease exclusively.
- Gradio becomes a client of this service instead of holding its own backend.
- The API surface exposes the same primitives internal callers already use
  (`encode_frame`, `predict_step`), plus a higher-level planning operation
  (`plan_step`), so it is useful as a real building block, not just an internal
  refactor of Gradio's plumbing.
- All device-touching work is serialized inside the service itself.
- Testable without hardware, the same way the rest of this repo already is.

## Non-goals (explicitly out of scope for this pass)

- Session-handle-based context caching (server keeps the growing `reps` tensor and the
  client only sends new actions). The first version sends the full context on every
  call, mirroring today's in-process call signature exactly. Payload size will grow
  with rollout length, same shape as the DRAM-growth issue already solved once for
  in-process rollouts — acceptable for now, revisit if it's a real cost once this is in
  use.
- Multi-tenant auth, rate limiting, or any deployment/ops concerns beyond running the
  service and pointing Gradio at it.
- No changes to `run_show`, `run_choreography`, `run_grounded_check`, or `run_cem_plan`
  themselves — they already take `backend` as a parameter and only depend on its duck
  type. The only code that changes is `main()`'s backend construction. Because every
  tab shares that one `backend` instance, switching to `RemoteBackend` means the Show
  tab goes through the service too, automatically — there's no way to route only some
  tabs through it. The manual smoke test (see Rollout plan) covers Grounded Check,
  Dance, and CEM Planning explicitly; Show is not separately exercised in this pass,
  though it will in fact be running over the new client like everything else.

## Architecture

A new top-level `service/` directory, sibling to `gradio_app/` and `tt/`:

```
service/
  app.py          # FastAPI app: endpoint definitions, request/response models
  worker.py       # single-worker serialization: one queue, one thread, owns the backend
  wire.py         # tensor <-> safetensors, frame <-> base64 PNG (de)serialization
  test_app.py     # endpoint tests via httpx.ASGITransport + the existing stub backend
```

`gradio_app/backends.py` gains a third class, `RemoteBackend`, implementing the same
duck-typed interface as `ReferenceBackend`/`TTNNBackend` (`.name`, `.encode_frame()`,
`.predict_step()`) by calling the service over HTTP. `gradio_app/app.py`'s `main()`
gains a `--backend remote --service-url ...` option alongside the existing
`ttnn`/`reference` choices; nothing in `planning.py` or the rollout functions changes,
since they only depend on the `backend` duck type.

### Why a single dedicated worker, not per-request async handlers

`encode_frame`/`predict_step` are synchronous, device-bound calls (`torch`/`ttnn`), not
async-native. Running them directly inside an async request handler would block the
event loop; running them via a naive threadpool would reintroduce exactly the
concurrent-device-access bug we just fixed in Gradio. The service instead runs a single
background thread that owns the backend and processes one request at a time from a
queue — every endpoint handler puts a request on the queue and awaits its result. This
is the same guarantee `concurrency_id="ttnn-backend", concurrency_limit=1` gave Gradio,
enforced once, centrally, for every client rather than only Gradio's own tabs.

## API

All endpoints are `POST`, JSON body except where noted. Every endpoint is single-session
(batch size 1) — the leading batch dimension `predict_step`/`plan_step` use in-process
is omitted on the wire: `reps` is `[T, HW, D]`, `actions`/`states`/`prior_actions`/
`states_seq` are `[T, 7]` (or `[7]` for a single pose/action like `cur_pose`/
`target_action`/`found_action`). The service adds the batch dimension back before
calling the backend and strips it again before responding.

### `POST /encode`
Stateless: one frame in, one embedding out.

- Request: `{"frame_png_b64": "<base64 PNG>"}`
- Response: `{"rep": "<base64 safetensors, one tensor named 'rep'>"}`

### `POST /predict_step`
Mirrors `backend.predict_step(reps, actions, states)` exactly — the caller owns its own
growing context, same as any in-process caller today.

- Request: `{"reps": "<base64 safetensors>", "actions": [[...7 floats...], ...], "states": [[...7 floats...], ...]}`
  (`actions`/`states` are small — `[T, 7]` — so they travel as plain JSON, not safetensors)
- Response: `{"next_rep": "<base64 safetensors>", "latency_ms": <float>}`

### `POST /plan_step`
Wraps `planning.plan_step`.

- Request: `{"reps": "<base64 safetensors>", "prior_actions": [[...7 floats...], ...], "states_seq": [[...7 floats...], ...], "cur_pose": [...7 floats...], "target_action": [...7 floats...], "cem_steps": 6, "samples": 12, "topk": 4, "maxnorm": 0.12}`
- Response: `{"found_action": [...7 floats...], "next_rep": "<base64 safetensors>", "energy": <float>, "latency_ms": <float>}`

### Errors
Any exception raised by the backend (shape mismatch, device error, etc.) is caught at
the worker boundary and returned as `HTTP 500` with `{"error": "<str(exception)>"}` —
no attempt to classify or recover from specific backend failures in this pass; the
caller sees the same exception text a direct in-process call would have raised.

## Wire format

- Tensors: `safetensors` (not pickle — no arbitrary code execution risk, and it's the
  standard for this exact purpose), base64-encoded for JSON embedding.
- Frames: base64-encoded PNG (`np.ndarray [H,W,3] uint8` <-> PNG round-trip, lossless).
- Everything else (actions, states, poses, scalars, CEM params): plain JSON.

## Testing

- `service/test_app.py`: endpoint-level tests against the FastAPI app via
  `httpx.ASGITransport` (in-process, no real socket, no hardware) — reuses the
  `_StubBackend`/`_RecordingStubBackend` pattern from `gradio_app/test_planning.py`.
  Covers: `/encode`, `/predict_step`, `/plan_step` round-trip correctness (response
  matches calling the stub backend directly with the same decoded inputs), and that the
  worker serializes concurrent requests (two requests submitted concurrently; the
  second's backend call does not start until the first's finishes — same style of
  proof used to verify the Gradio `concurrency_id` fix, via the stub's call-order log
  rather than `py-spy` this time since the service is a much smaller, controllable
  surface).
- `gradio_app/test_backends.py` (new): `RemoteBackend` against a real running instance
  of the FastAPI app (via `httpx.ASGITransport` again, no real network) with the stub
  backend underneath — proves the client-side serialization/deserialization round-trips
  correctly.
- No new hardware-dependent tests in this pass; a manual real-backend smoke test
  (service running with `--backend ttnn`, Gradio pointed at it with `--backend remote`)
  is the acceptance check before calling this done, same as the last two features.

## Rollout plan

1. `service/wire.py` — tensor/frame (de)serialization helpers, unit-tested directly
   (no FastAPI needed for these).
2. `service/worker.py` — the single-worker queue, tested with the stub backend
   (proves serialization without any HTTP layer).
3. `service/main.py` — FastAPI endpoints wrapping the worker, tested via
   `httpx.ASGITransport`.
4. `gradio_app/backends.py::RemoteBackend` — HTTP client implementing the backend duck
   type, tested against the FastAPI app in-process.
5. `gradio_app/app.py` — add `--backend remote --service-url`, wire it to
   `RemoteBackend`.
6. Manual real-backend smoke test: run the service with `--backend ttnn` under a gozer
   lease, run Gradio with `--backend remote`, exercise Grounded Check / Dance /
   CEM Planning tabs.
