# SPDX-License-Identifier: MIT
"""A small library of named 7-DoF action primitives ("moves"), each a short
(dx,dy,dz,droll,dpitch,dyaw,dgripper) sequence -- the building blocks a choreography
chains together. Magnitudes are kept within the same range CEM sampling uses in Meta's
own mpc_utils.py (maxnorm=0.05 translation/step, gripper clipped to [-0.75, 0.75]) --
these aren't validated against a real robot's dynamics, just kept physically plausible
in scale.

Every move is a pure function `(n_steps) -> np.ndarray[n_steps, 7]`; MOVES maps a
display name to one. A choreography is just a list of move names, whose action arrays
get concatenated and fed through the same imagination-rollout loop `run_dance` already
uses -- no new model-side machinery, only more organized action generation.
"""

import numpy as np


def move_wave(n_steps: int, amp: float = 0.035) -> np.ndarray:
    """WELLE -- a side-to-side sway with a light vertical bob, gripper neutral."""
    t = np.arange(n_steps)
    dy = amp * np.sin(t / max(n_steps - 1, 1) * 2 * np.pi)
    dz = amp * 0.3 * np.sin(t / max(n_steps - 1, 1) * 4 * np.pi)
    zeros = np.zeros(n_steps)
    return np.stack([zeros, dy, dz, zeros, zeros, zeros, zeros], axis=1).astype(np.float32)


def move_spin(n_steps: int, total_yaw: float = 1.1) -> np.ndarray:
    """SPIN -- a smooth partial rotation about yaw, minimal translation."""
    dyaw = np.full(n_steps, total_yaw / n_steps)
    zeros = np.zeros(n_steps)
    return np.stack([zeros, zeros, zeros, zeros, zeros, dyaw, zeros], axis=1).astype(np.float32)


def move_bow(n_steps: int, depth: float = 0.045, tilt: float = 0.35) -> np.ndarray:
    """VERBEUGUNG (bow) -- dip down and pitch forward, then rise back, symmetric
    about the midpoint."""
    half = n_steps // 2
    phase = np.concatenate([np.linspace(0, 1, half), np.linspace(1, 0, n_steps - half)])
    dz = -depth * np.diff(np.concatenate([[0], phase]))
    dpitch = tilt * np.diff(np.concatenate([[0], phase]))
    zeros = np.zeros(n_steps)
    return np.stack([zeros, zeros, dz, zeros, dpitch, zeros, zeros], axis=1).astype(np.float32)


def move_snap(n_steps: int) -> np.ndarray:
    """SCHNAPP (snap) -- rapid gripper open/close, negligible motion otherwise."""
    dgripper = np.where(np.arange(n_steps) % 2 == 0, 0.3, -0.3)
    zeros = np.zeros(n_steps)
    return np.stack([zeros, zeros, zeros, zeros, zeros, zeros, dgripper], axis=1).astype(np.float32)


def move_figure_eight(n_steps: int, amp: float = 0.04) -> np.ndarray:
    """ACHT (figure-eight) -- a Lissajous sway in (x,y) with a gentle z bob (the
    original single-parametric dance, now one move among several)."""
    t = np.arange(n_steps)
    dx = amp * np.sin(t / n_steps * 2 * np.pi)
    dy = amp * np.sin(2 * t / n_steps * 2 * np.pi) * 0.6
    dz = amp * 0.5 * np.cos(t / n_steps * 2 * np.pi)
    zeros = np.zeros(n_steps)
    return np.stack([dx, dy, dz, zeros, zeros, zeros, zeros], axis=1).astype(np.float32)


MOVES = {
    "WELLE": move_wave,
    "SPIN": move_spin,
    "VERBEUGUNG": move_bow,
    "SCHNAPP": move_snap,
    "ACHT": move_figure_eight,
}


def build_choreography(sequence: str, steps_per_move: int) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """`sequence` is a whitespace-separated list of move names (repeats allowed,
    unknown names skipped). Returns (concatenated actions [N,7], segments), where
    segments is [(name, start_step, end_step), ...] for reporting/labeling."""
    names = [tok.strip().upper() for tok in sequence.split() if tok.strip()]
    segments = []
    chunks = []
    cursor = 0
    for name in names:
        move_fn = MOVES.get(name)
        if move_fn is None:
            continue
        chunk = move_fn(steps_per_move)
        chunks.append(chunk)
        segments.append((name, cursor, cursor + steps_per_move))
        cursor += steps_per_move
    if not chunks:
        return np.zeros((0, 7), dtype=np.float32), []
    return np.concatenate(chunks, axis=0), segments
