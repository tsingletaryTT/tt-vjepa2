# SPDX-License-Identifier: MIT
"""Tests for app.py's CEM-driven imagination rollout (cem_imagination_rollout) and the
run_choreography use_planning toggle that dispatches to it. No hardware needed: reuses
test_planning's deterministic stub backend."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import cem_imagination_rollout, imagination_rollout, run_choreography  # noqa: E402
from test_planning import _StubBackend  # noqa: E402


def test_cem_imagination_rollout_matches_shape_of_imagination_rollout():
    hw, d = 2, 3
    backend = _StubBackend(hw, d)
    rep0 = torch.zeros(1, hw, d)
    start_pose = np.zeros(7, dtype=np.float32)
    target_actions = np.array(
        [
            [0.02, 0.0, 0.0, 0, 0, 0, 0.0],
            [0.0, 0.03, 0.0, 0, 0, 0, 0.1],
            [-0.01, 0.0, 0.02, 0, 0, 0, -0.1],
        ],
        dtype=np.float32,
    )

    torch.manual_seed(3)
    poses, energies, latencies = cem_imagination_rollout(
        backend, start_frame=None, start_pose=start_pose, target_actions=target_actions, rep0=rep0,
        cem_steps=2, samples=4, topk=2, maxnorm=0.1,
    )

    assert poses.shape == (len(target_actions) + 1, 7)
    assert energies.shape == (len(target_actions),)
    assert latencies.shape == (len(target_actions),)
    np.testing.assert_allclose(poses[0], start_pose)


def test_cem_imagination_rollout_executes_planned_not_target_actions():
    """The rendered trajectory must come from what CEM actually found at each step,
    not target_actions played back verbatim -- otherwise 'plan with CEM' would be
    indistinguishable from the scripted path it's supposed to replace."""
    hw, d = 2, 3
    backend = _StubBackend(hw, d)
    rep0 = torch.zeros(1, hw, d)
    start_pose = np.zeros(7, dtype=np.float32)
    target_actions = np.array([[0.09, -0.07, 0.05, 0, 0, 0, 0.2]], dtype=np.float32)

    torch.manual_seed(11)
    planned_poses, _, _ = cem_imagination_rollout(
        backend, start_frame=None, start_pose=start_pose, target_actions=target_actions, rep0=rep0,
        cem_steps=3, samples=6, topk=2, maxnorm=0.1,
    )

    scripted_poses, _, _ = imagination_rollout(
        backend, start_frame=None, start_pose=start_pose, actions=target_actions, rep0=rep0
    )

    assert not np.allclose(planned_poses[1], scripted_poses[1])


def test_run_choreography_use_planning_toggle_dispatches_to_cem_rollout():
    """use_planning=True must route through cem_imagination_rollout (many backend
    calls per step: a goal probe + CEM's iterations + an execute), not
    imagination_rollout (exactly one backend call per step) -- checked by call count,
    since run_choreography's return (a plotly figure + markdown) doesn't expose the
    underlying poses directly."""
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    states = np.zeros((1, 7), dtype=np.float32)

    backend_direct = _StubBackend(2, 3)
    torch.manual_seed(0)
    fig_direct, report_direct = run_choreography(backend_direct, frames, states, "SPIN", steps_per_move=2,
                                                  use_planning=False)

    backend_planned = _StubBackend(2, 3)
    torch.manual_seed(0)
    fig_planned, report_planned = run_choreography(backend_planned, frames, states, "SPIN", steps_per_move=2,
                                                    use_planning=True)

    assert len(backend_direct.calls) == 2  # one predict_step call per step, scripted
    assert len(backend_planned.calls) > len(backend_direct.calls)
    assert "Plan with CEM" not in report_direct
    assert "Plan with CEM" in report_planned
