# SPDX-License-Identifier: MIT
"""Routes backend calls onto a single dedicated thread, so concurrent requests are
serialized against the (synchronous, device-bound) backend instead of racing each
other -- the same guarantee gradio_app/app.py's concurrency_id="ttnn-backend",
concurrency_limit=1 gives Gradio's own tabs, enforced here centrally so it protects
any client, not just Gradio. A ThreadPoolExecutor(max_workers=1) already IS a single
dedicated worker thread with its own FIFO work queue -- no need to hand-roll one.
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradio_app"))
from planning import plan_step as plan_step_fn  # noqa: E402


class BackendWorker:
    def __init__(self, backend):
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def encode_frame(self, frame_uint8):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.backend.encode_frame, frame_uint8)

    async def predict_step(self, reps, actions, states):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.backend.predict_step, reps, actions, states)

    async def plan_step(self, reps, prior_actions, states_seq, cur_pose, target_action, **cem_kwargs):
        loop = asyncio.get_running_loop()

        def call():
            return plan_step_fn(self.backend, reps, prior_actions, states_seq, cur_pose, target_action, **cem_kwargs)

        return await loop.run_in_executor(self._executor, call)

    def shutdown(self):
        self._executor.shutdown(wait=True)
