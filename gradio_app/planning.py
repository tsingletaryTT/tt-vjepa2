# SPDX-License-Identifier: MIT
"""A faithful reimplementation of Meta's own CEM (Cross-Entropy Method) action search
from reference/notebooks/utils/mpc_utils.py::cem, adapted to call our backend
abstraction's `predict_step` instead of their notebook's vectorized world_model
wrapper -- their `cem()` expects a differently-shaped callable tied directly to their
raw encoder/predictor objects, so this ports the algorithm (Gaussian sampling ->
evaluate -> keep top-k -> momentum-blend mean/std -> repeat), not just calls their
function. Like their own cem(), all samples for one iteration are batched through a
single predict_step call (the sample dimension is the backend's batch dimension), not
looped one at a time.

Matches their own simplification: only translation (dx,dy,dz) and gripper are
searched, rotation is held at zero (see their `sample_action_traj`, which builds each
sample as `cat([xyz, zeros(3), gripper])`). Our real recorded action between the demo's
two frames has a small but nonzero rotation component, so CEM here converges toward the
best *rotation-locked* approximation of it, not an exact match -- reported honestly
rather than papered over.
"""

import numpy as np
import torch


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).sqrt())


def l2_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row counterpart to l2: a is [S, ...], b broadcasts against it (typically
    [1, ...], one shared goal). Returns [S], the L2 error of each row against b."""
    dims = tuple(range(1, a.dim()))
    return torch.sqrt(torch.mean((a - b) ** 2, dim=dims))


def cem_search(
    backend,
    rep0: torch.Tensor,
    pose0: np.ndarray,
    goal_rep: torch.Tensor,
    cem_steps: int = 6,
    samples: int = 12,
    topk: int = 4,
    maxnorm: float = 0.05,
    gripper_std: float = 0.3,
    momentum_mean: float = 0.25,
    momentum_std: float = 0.6,
):
    """Returns (final_action [7], history) where history is a list of per-iteration
    {step, best_error, mean_error, mean_action} dicts -- everything needed to plot
    convergence and report the final result."""
    reps = rep0.unsqueeze(1)  # [1,1,HW,D]
    pose0_t = torch.from_numpy(pose0).float().view(1, 1, 7)

    # Every sample shares the same starting context/pose -- repeat both across the
    # sample-batch dimension once, up front, rather than per iteration. Mirrors Meta's
    # own reference cem()'s `context_frame.repeat(samples, ...)`.
    reps_batch = reps.expand(samples, *reps.shape[1:])
    pose_batch = pose0_t.expand(samples, *pose0_t.shape[1:])

    mean = torch.zeros(4)  # dx, dy, dz, dgripper
    std = torch.tensor([maxnorm, maxnorm, maxnorm, gripper_std])
    history = []

    for step in range(cem_steps):
        candidates = torch.randn(samples, 4) * std + mean
        candidates[:, :3] = torch.clip(candidates[:, :3], -maxnorm, maxnorm)
        candidates[:, 3:4] = torch.clip(candidates[:, 3:4], -0.75, 0.75)

        actions7 = torch.zeros(samples, 7)
        actions7[:, :3] = candidates[:, :3]
        actions7[:, 6] = candidates[:, 3]
        actions_t = actions7.view(samples, 1, 7)
        rep_pred, _ = backend.predict_step(reps_batch, actions_t, pose_batch)
        errors = l2_batched(rep_pred, goal_rep)

        idx = errors.topk(topk, largest=False).indices
        top = candidates[idx]
        new_mean = top.mean(dim=0)
        new_std = top.std(dim=0)
        mean = new_mean * (1 - momentum_mean) + mean * momentum_mean
        std = new_std * (1 - momentum_std) + std * momentum_std

        best_idx = errors.argmin()
        history.append(
            {
                "step": step,
                "best_error": float(errors.min()),
                "mean_error": float(errors.mean()),
                "best_candidate": candidates[best_idx, :4].tolist(),
            }
        )

    final_action = torch.zeros(7)
    final_action[:3] = mean[:3]
    final_action[6] = mean[3]
    return final_action.numpy(), history


def plan_step(
    backend,
    reps: torch.Tensor,
    prior_actions: list,
    states_seq: list,
    cur_pose: np.ndarray,
    target_action: np.ndarray,
    cem_steps: int = 6,
    samples: int = 12,
    topk: int = 4,
    maxnorm: float = 0.12,
):
    """One goal-directed planning step, replacing "execute this hand-authored action"
    with "search for a real action reaching roughly the same intent":

    1. Probe a goal embedding by running `target_action` (the intended delta -- e.g.
       one step of a moves.py primitive) through the REAL growing context (reps plus
       the full prior_actions/states_seq history, matching how predict_step is used
       everywhere else in the app).
    2. Let CEM search (a local, single-step lookahead from the current latest frame --
       same design the CEM Planning tab already uses) find an action that actually
       reaches near that goal.
    3. Execute the FOUND action -- never the target -- through that same real context.
       This is the actually-rendered step: the model chose it, the target only ever
       supplied intent for the probe.

    reps: [1,T,HW,D], the context already realized. prior_actions/states_seq: lists of
    length T-1 and T respectively (states_seq[-1] == cur_pose, the pose the new action
    is taken from) -- same convention imagination_rollout already uses. Does not
    mutate its inputs.

    Returns (found_action [7] np.ndarray, next_rep [1,HW,D] tensor, energy float,
    latency_ms float) -- energy is next_rep's L2 distance from the context's current
    last frame, matching imagination_rollout's own per-step signal.
    """
    states_t = torch.from_numpy(np.stack(states_seq)).float().unsqueeze(0)

    probe_actions = prior_actions + [target_action]
    probe_actions_t = torch.from_numpy(np.stack(probe_actions)).float().unsqueeze(0)
    goal_rep, _ = backend.predict_step(reps, probe_actions_t, states_t)

    found_action, _history = cem_search(
        backend, reps[:, -1], cur_pose, goal_rep,
        cem_steps=cem_steps, samples=samples, topk=topk, maxnorm=maxnorm,
    )

    exec_actions = prior_actions + [found_action]
    exec_actions_t = torch.from_numpy(np.stack(exec_actions)).float().unsqueeze(0)
    next_rep, latency_ms = backend.predict_step(reps, exec_actions_t, states_t)
    energy = l2(next_rep, reps[:, -1])
    return found_action, next_rep, energy, latency_ms
