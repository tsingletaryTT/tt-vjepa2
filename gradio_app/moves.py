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
    """WAVE -- a side-to-side sway with a light vertical bob, gripper neutral."""
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
    """BOW -- dip down and pitch forward, then rise back, symmetric
    about the midpoint."""
    half = n_steps // 2
    phase = np.concatenate([np.linspace(0, 1, half), np.linspace(1, 0, n_steps - half)])
    dz = -depth * np.diff(np.concatenate([[0], phase]))
    dpitch = tilt * np.diff(np.concatenate([[0], phase]))
    zeros = np.zeros(n_steps)
    return np.stack([zeros, zeros, dz, zeros, dpitch, zeros, zeros], axis=1).astype(np.float32)


def move_snap(n_steps: int) -> np.ndarray:
    """SNAP -- rapid gripper open/close, negligible motion otherwise."""
    dgripper = np.where(np.arange(n_steps) % 2 == 0, 0.3, -0.3)
    zeros = np.zeros(n_steps)
    return np.stack([zeros, zeros, zeros, zeros, zeros, zeros, dgripper], axis=1).astype(np.float32)


def move_figure_eight(n_steps: int, amp: float = 0.04) -> np.ndarray:
    """FIGURE8 (figure-eight) -- a Lissajous sway in (x,y) with a gentle z bob (the
    original single-parametric dance, now one move among several)."""
    t = np.arange(n_steps)
    dx = amp * np.sin(t / n_steps * 2 * np.pi)
    dy = amp * np.sin(2 * t / n_steps * 2 * np.pi) * 0.6
    dz = amp * 0.5 * np.cos(t / n_steps * 2 * np.pi)
    zeros = np.zeros(n_steps)
    return np.stack([dx, dy, dz, zeros, zeros, zeros, zeros], axis=1).astype(np.float32)


def move_freeze(n_steps: int) -> np.ndarray:
    """FREEZE (lock) -- hold the current pose exactly, zero action every step.
    Not a move on its own so much as punctuation between them (the "lock" half of
    Poppin and Lockin)."""
    return np.zeros((n_steps, 7), dtype=np.float32)


MOVES = {
    "WAVE": move_wave,
    "SPIN": move_spin,
    "BOW": move_bow,
    "SNAP": move_snap,
    "FIGURE8": move_figure_eight,
    "FREEZE": move_freeze,
}


# --- Letter tracing (the "XOXO TT" finale) -------------------------------------------
#
# Each letter is a set of strokes in a local 0..1 x 0..1 unit square. The move TO a
# stroke's first waypoint is a pen-up jump (gripper opens, trail breaks there); moves
# WITHIN a stroke are pen-down draws (gripper closes, trail connects). This is
# necessarily a gestural, low-resolution trace -- a handful of straight-line segments
# from a stylized 2-link arm, not literal handwriting -- so the HUD annotation names
# the letter in text alongside the physical attempt, rather than relying on the shape
# alone to read as a letter.
LETTER_STROKES = {
    "X": [[(0, 1), (1, 0)], [(0, 0), (1, 1)]],
    "O": [[(0.5, 1.0), (1.0, 0.5), (0.5, 0.0), (0.0, 0.5)]],  # a diamond, not a circle --
    # kept to 4 points (not the rounder ~7-8 it takes to read as an actual circle)
    # because each additional point is one more distinct growing-context sequence
    # length the TTNN backend has to allocate attention buffers for; found the hard
    # way that this rollout's full (non-windowed) self-attention makes memory grow
    # with the SQUARE of total steps, so every point here is a real, not free, cost
    # (see app.py's run_show docstring / moves.build_show for the actual ceiling hit).
    "T": [[(0, 1), (1, 1)], [(0.5, 1), (0.5, 0)]],
}


def _letter_deltas(letter: str, scale: float, advance: tuple = (1.3, 0.0)):
    """Returns (deltas [n,2] d-advance,d-up, pen_up [n] bool) for one letter, plus the
    implicit final advance jump to the next letter's local origin. `letter=' '` is just
    the advance with no strokes (a gap)."""
    strokes = LETTER_STROKES.get(letter, [])
    pos = np.zeros(2, dtype=np.float32)
    deltas, pen_up = [], []
    for stroke in strokes:
        for i, (wx, wy) in enumerate(stroke):
            target = np.array([wx, wy], dtype=np.float32)
            deltas.append((target - pos) * scale)
            pen_up.append(i == 0)  # the jump TO a stroke's start is pen-up
            pos = target
    target = np.array(advance, dtype=np.float32)
    deltas.append((target - pos) * scale)  # `pos` is still in unscaled unit-square space
    pen_up.append(True)
    return np.array(deltas, dtype=np.float32), np.array(pen_up, dtype=bool)


def build_letters(text: str, scale: float = 0.03):
    """text, e.g. 'XOXO TT' -- returns (actions [N,7], pen_up [N] bool, labels [N] str)
    where labels[i] names which letter step i belongs to (for the HUD). Gripper closes
    a little on pen-down draws, opens a little on pen-up jumps -- the existing
    gripper-driven marker size in robot_viz.py doubles as a "pen lifted" cue for free."""
    all_actions, all_pen_up, all_labels = [], [], []
    for ch in text.upper():
        deltas, pen_up = _letter_deltas(ch, scale)
        n = len(deltas)
        actions = np.zeros((n, 7), dtype=np.float32)
        # Trace in the world X-Z plane (advance = +x, "up" in the letter = +z), not
        # X-Y (the horizontal floor plane) -- letters need a vertical "whiteboard"
        # orientation to read as writing at all; traced flat on the horizontal plane,
        # a 3D camera at any elevated angle sees them foreshortened and, depending on
        # which side it's on, mirrored (found this the hard way: it read right-to-left
        # and upside down).
        actions[:, 0] = deltas[:, 0]  # dx: advance / reading direction
        actions[:, 2] = deltas[:, 1]  # dz: up in the letter = up in the world
        actions[:, 6] = np.where(pen_up, 0.12, -0.12)  # open on jumps, close on draws
        all_actions.append(actions)
        all_pen_up.append(pen_up)
        all_labels.extend([ch if ch != " " else "·"] * n)
    return np.concatenate(all_actions, axis=0), np.concatenate(all_pen_up, axis=0), all_labels


def move_axe_swing(n_steps: int = 2, amp: float = 0.09) -> np.ndarray:
    """AXE_SWING -- a single, sudden downward-forward strike (the "Ax" in Careful with
    that Ax, Eugene): one big step, gripper snapping shut on impact then back open."""
    dx, dz, dgripper = (np.zeros(n_steps) for _ in range(3))
    dx[0] = amp * 0.6
    dz[0] = -amp
    dgripper[0] = -0.6
    if n_steps > 1:
        dgripper[1] = 0.6
    zeros = np.zeros(n_steps)
    return np.stack([dx, zeros, dz, zeros, zeros, zeros, dgripper], axis=1).astype(np.float32)


def build_show():
    """The fixed multi-act show (see app.py's auto-looping player): a produced arc,
    not a per-play-random generator -- 'always evolving' here means a real six-plus-one
    act performance that plays start to finish and loops, each act with its own
    palette/camera framing (interpreted by robot_viz.py using the metadata below), not
    a single continuous wiggle. Returns (actions [N,7], pen_up [N] bool, labels [N] str,
    act_segments [{name,start,end,palette,zoom,camera_bias,speed}, ...])."""
    acts = []

    def add(name, actions, palette, zoom, camera_bias, pen_up=None, labels=None, loops=1, speed=1.0):
        # Repeating a short motion (rather than authoring one long pass) is how each
        # act gets to linger without growing the show's total step count much -- and
        # total step count is a real budget here, not a style choice: each distinct
        # sequence length costs its own one-time TTNN kernel compile, and the growing
        # imagination-rollout context makes memory grow with the SQUARE of total
        # steps (see functional_predictor.py's get_rope_and_mask docstring) -- this
        # show already found that ceiling once. Most of these moves are themselves
        # roughly periodic (sin-based, or symmetric-phase), so repeating them reads as
        # "doing the move again", not a drift away from where it started.
        #
        # `speed`: playback-speed multiplier (>1 faster, <1 slower) -- purely a
        # presentation choice for robot_viz.py's per-frame duration, not something
        # that changes the actual rollout. A millimeter-scale act (Grain of Rice)
        # needs a slower frame duration or its gesture reads as a single blink.
        actions = np.tile(actions, (loops, 1)).astype(np.float32)
        n = len(actions)
        if pen_up is not None:
            pen_up = np.tile(pen_up, loops)
        if labels is not None:
            labels = list(labels) * loops
        acts.append(
            dict(
                name=name,
                actions=actions,
                pen_up=pen_up if pen_up is not None else np.zeros(n, dtype=bool),
                labels=labels if labels is not None else [name] * n,
                palette=palette,
                zoom=zoom,
                camera_bias=camera_bias,
                speed=speed,
            )
        )

    add(
        "INSTANT KRAFTWERK",
        np.concatenate([move_figure_eight(2), move_spin(2, total_yaw=0.4)]),
        palette="red",
        zoom=1.7,
        camera_bias=(1.0, 1.0, 0.6),
        loops=2,
    )

    add(
        "CAREFUL WITH THAT AX, EUGENE",
        move_bow(3, depth=0.02, tilt=0.15),
        palette="cool",
        zoom=2.1,
        camera_bias=(0.3, 1.8, 0.3),
    )
    add(
        "CAREFUL WITH THAT AX, EUGENE",
        move_axe_swing(2),
        palette="flash",
        zoom=1.1,
        camera_bias=(1.8, 0.3, 1.4),
        loops=2,
    )

    add(
        "TANGERINE RATCHET",
        np.concatenate([move_spin(1, total_yaw=0.3), move_snap(1)]),
        palette="tangerine",
        zoom=1.5,
        camera_bias=(1.6, 1.6, 0.2),
        loops=3,
    )

    add(
        "POPPIN AND LOCKIN",
        np.concatenate([move_snap(1), move_freeze(1)]),
        palette="red",
        zoom=1.4,
        camera_bias=(0.2, 1.8, 0.9),
        loops=3,
    )

    add(
        "YOUR NAME ON A GRAIN OF RICE",
        move_wave(2, amp=0.006),
        palette="micro",
        zoom=0.55,
        camera_bias=(0.4, 0.4, 0.25),
        loops=2,
        speed=0.4,
    )

    add(
        "LASER CATS CUTTING A RUG",
        np.concatenate([move_figure_eight(2, amp=0.05), move_spin(2, total_yaw=0.6)]),
        palette="rainbow",
        zoom=1.3,
        camera_bias=(1.9, 0.6, 1.1),
        loops=2,
    )

    letters_actions, letters_pen_up, letters_labels = build_letters("XOXO TT")
    # Letters are traced in the world X-Z plane (see build_letters) -- the camera needs
    # to face that plane roughly along Y, and specifically from -Y looking toward +Y,
    # for the advance direction (+x) to read as screen-right rather than mirrored.
    add(
        "XOXO TT",
        letters_actions,
        palette="rainbow",
        zoom=1.5,
        camera_bias=(0.2, -1.7, 0.2),
        pen_up=letters_pen_up,
        labels=letters_labels,
    )

    all_actions = np.concatenate([a["actions"] for a in acts], axis=0)
    all_pen_up = np.concatenate([a["pen_up"] for a in acts], axis=0)
    all_labels = sum((a["labels"] for a in acts), [])
    act_segments = []
    cursor = 0
    for a in acts:
        n = len(a["actions"])
        act_segments.append(
            dict(
                name=a["name"],
                start=cursor,
                end=cursor + n,
                palette=a["palette"],
                zoom=a["zoom"],
                camera_bias=a["camera_bias"],
                speed=a["speed"],
            )
        )
        cursor += n
    return all_actions, all_pen_up, all_labels, act_segments


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
