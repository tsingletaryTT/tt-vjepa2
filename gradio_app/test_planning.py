# SPDX-License-Identifier: MIT
"""Correctness tests for planning.py's CEM search.

cem_search's sample loop originally called backend.predict_step once per sample, in a
Python for-loop -- serial, and each call pays a full backend round-trip. This batches
all samples for one CEM iteration into a single predict_step call (matching how Meta's
own reference `cem()` already batches the sample dimension). The batched version must
produce bit-identical results to the original per-sample-loop algorithm given the same
RNG seed (same candidate sampling, same math, just fewer backend calls) -- that
equivalence, plus the call-count itself, is what these tests verify. No hardware
needed: a deterministic stub stands in for the backend.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planning import cem_search, l2, l2_batched, plan_step  # noqa: E402


class _StubBackend:
    """Deterministic, cheap stand-in for a real backend: predict_step's output depends
    only on the mean of the given action(s), broadcast over (HW, D) -- enough for CEM's
    search to converge reproducibly, cheap enough to run instantly. Records every
    call's batch size so tests can assert on how many backend calls a search made."""

    name = "stub"

    def __init__(self, hw: int = 2, d: int = 3):
        self.hw, self.d = hw, d
        self.calls: list[int] = []

    def predict_step(self, reps, actions, states):
        B = actions.shape[0]
        self.calls.append(B)
        base = actions.mean(dim=(1, 2))  # [B]
        next_rep = base.view(B, 1, 1).expand(B, self.hw, self.d).clone()
        return next_rep, 0.0

    def encode_frame(self, frame_uint8):
        return torch.zeros(1, self.hw, self.d)


def _reference_cem_search_unbatched(backend, rep0, pose0, goal_rep, cem_steps, samples, topk,
                                     maxnorm, gripper_std=0.3, momentum_mean=0.25, momentum_std=0.6):
    """Independent re-derivation of the pre-batching algorithm: one predict_step call
    per sample, in a loop. The oracle a batched implementation must match bit-for-bit
    given the same RNG seed and the same (stub) backend."""
    reps = rep0.unsqueeze(1)
    pose0_t = torch.from_numpy(pose0).float().view(1, 1, 7)
    mean = torch.zeros(4)
    std = torch.tensor([maxnorm, maxnorm, maxnorm, gripper_std])

    for _ in range(cem_steps):
        candidates = torch.randn(samples, 4) * std + mean
        candidates[:, :3] = torch.clip(candidates[:, :3], -maxnorm, maxnorm)
        candidates[:, 3:4] = torch.clip(candidates[:, 3:4], -0.75, 0.75)

        errors = torch.zeros(samples)
        for s in range(samples):
            action7 = torch.zeros(7)
            action7[:3] = candidates[s, :3]
            action7[6] = candidates[s, 3]
            rep_pred, _ = backend.predict_step(reps, action7.view(1, 1, 7), pose0_t)
            errors[s] = l2(rep_pred, goal_rep)

        idx = errors.topk(topk, largest=False).indices
        top = candidates[idx]
        mean = top.mean(dim=0) * (1 - momentum_mean) + mean * momentum_mean
        std = top.std(dim=0) * (1 - momentum_std) + std * momentum_std

    final_action = torch.zeros(7)
    final_action[:3] = mean[:3]
    final_action[6] = mean[3]
    return final_action.numpy()


def test_l2_batched_matches_per_row_l2():
    a = torch.randn(5, 4, 6)
    b = torch.randn(1, 4, 6)
    batched = l2_batched(a, b)
    assert batched.shape == (5,)
    for i in range(5):
        assert torch.isclose(batched[i], torch.tensor(l2(a[i : i + 1], b)), atol=1e-6)


def test_cem_search_matches_unbatched_reference_given_same_seed():
    hw, d = 2, 3
    rep0 = torch.zeros(1, hw, d)
    pose0 = np.zeros(7, dtype=np.float32)
    goal_rep = torch.ones(1, hw, d) * 0.05
    cem_steps, samples, topk, maxnorm = 3, 6, 2, 0.1

    torch.manual_seed(42)
    expected = _reference_cem_search_unbatched(
        _StubBackend(hw, d), rep0, pose0, goal_rep, cem_steps, samples, topk, maxnorm
    )

    torch.manual_seed(42)
    actual, _history = cem_search(
        _StubBackend(hw, d), rep0, pose0, goal_rep, cem_steps=cem_steps, samples=samples,
        topk=topk, maxnorm=maxnorm,
    )

    assert torch.allclose(torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-6)


def test_cem_search_calls_backend_once_per_iteration_not_once_per_sample():
    hw, d = 2, 3
    rep0 = torch.zeros(1, hw, d)
    pose0 = np.zeros(7, dtype=np.float32)
    goal_rep = torch.ones(1, hw, d) * 0.05
    cem_steps, samples = 4, 8

    stub = _StubBackend(hw, d)
    torch.manual_seed(0)
    cem_search(stub, rep0, pose0, goal_rep, cem_steps=cem_steps, samples=samples, topk=3, maxnorm=0.1)

    assert len(stub.calls) == cem_steps
    assert all(b == samples for b in stub.calls)


class _RecordingStubBackend(_StubBackend):
    """_StubBackend that also records each call's exact actions/states tensors, so a
    caller's wiring (which action/state sequence it actually constructed) can be
    checked directly instead of only inferred from batch size."""

    def __init__(self, hw: int = 2, d: int = 3):
        super().__init__(hw, d)
        self.call_args: list[tuple[torch.Tensor, torch.Tensor]] = []

    def predict_step(self, reps, actions, states):
        self.call_args.append((actions.clone(), states.clone()))
        return super().predict_step(reps, actions, states)


def test_plan_step_probes_goal_with_target_then_executes_cem_found_action():
    """plan_step's contract: (1) probe a goal by running prior_actions + target_action
    through the real context, (2) let CEM search for a real action from there, (3)
    execute the FOUND action (not the target) through that same context -- never the
    target itself, since the target is only ever used to define intent for the probe."""
    hw, d = 2, 3
    backend = _RecordingStubBackend(hw, d)
    reps = torch.arange(2 * hw * d, dtype=torch.float32).reshape(1, 2, hw, d)  # T=2
    prior_actions = [np.array([0.01, 0.0, 0.0, 0, 0, 0, 0.0], dtype=np.float32)]
    states_seq = [np.zeros(7, dtype=np.float32), np.array([0.01, 0, 0, 0, 0, 0, 0], dtype=np.float32)]
    cur_pose = states_seq[-1]
    target_action = np.array([0.05, -0.02, 0.03, 0, 0, 0, 0.1], dtype=np.float32)
    cem_steps, samples = 2, 4

    torch.manual_seed(7)
    found_action, next_rep, energy, latency_ms = plan_step(
        backend, reps, prior_actions, states_seq, cur_pose, target_action,
        cem_steps=cem_steps, samples=samples, topk=2, maxnorm=0.1,
    )

    # 1 goal-probe call + cem_steps batched CEM-iteration calls + 1 execute call.
    assert len(backend.call_args) == cem_steps + 2

    expected_states = torch.from_numpy(np.stack(states_seq)).float().unsqueeze(0)
    probe_actions, probe_states = backend.call_args[0]
    expected_probe_actions = torch.from_numpy(np.stack(prior_actions + [target_action])).float().unsqueeze(0)
    assert torch.allclose(probe_actions, expected_probe_actions)
    assert torch.allclose(probe_states, expected_states)

    exec_actions, exec_states = backend.call_args[-1]
    expected_exec_actions = torch.from_numpy(np.stack(prior_actions + [found_action])).float().unsqueeze(0)
    assert torch.allclose(exec_actions, expected_exec_actions, atol=1e-6)
    assert torch.allclose(exec_states, expected_states)
    # The found action must actually differ from the target -- it's CEM's own search
    # result, not a pass-through (proves step 3 doesn't silently execute the target).
    assert not np.allclose(found_action, target_action)

    expected_next_rep, expected_latency_ms = _StubBackend(hw, d).predict_step(reps, exec_actions, exec_states)
    assert torch.allclose(next_rep, expected_next_rep)
    assert latency_ms == expected_latency_ms
    assert energy == pytest.approx(l2(next_rep, reps[:, -1]), abs=1e-6)
