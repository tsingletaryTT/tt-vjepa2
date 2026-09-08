# SPDX-License-Identifier: MIT
"""A faithful reimplementation of Meta's own CEM (Cross-Entropy Method) action search
from reference/notebooks/utils/mpc_utils.py::cem, adapted to call our backend
abstraction's `predict_step` (one sample at a time, in a Python loop) instead of their
notebook's vectorized world_model wrapper -- their `cem()` expects a differently-shaped
callable tied directly to their raw encoder/predictor objects, so this ports the
algorithm (Gaussian sampling -> evaluate -> keep top-k -> momentum-blend mean/std ->
repeat), not just calls their function.

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

    mean = torch.zeros(4)  # dx, dy, dz, dgripper
    std = torch.tensor([maxnorm, maxnorm, maxnorm, gripper_std])
    history = []

    for step in range(cem_steps):
        candidates = torch.randn(samples, 4) * std + mean
        candidates[:, :3] = torch.clip(candidates[:, :3], -maxnorm, maxnorm)
        candidates[:, 3:4] = torch.clip(candidates[:, 3:4], -0.75, 0.75)

        errors = torch.zeros(samples)
        for s in range(samples):
            action7 = torch.zeros(7)
            action7[:3] = candidates[s, :3]
            action7[6] = candidates[s, 3]
            actions_t = action7.view(1, 1, 7)
            rep_pred, _ = backend.predict_step(reps, actions_t, pose0_t)
            errors[s] = l2(rep_pred, goal_rep)

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
