# SPDX-License-Identifier: MIT
"""Regression tests for build_multi_dancer_figure.

History: a first version put every dancer on one shared 3D stage with a single
fixed camera. That hit two separate bugs (only the first two dancers ever
animated, then a subtler initial-frame trace-indexing bug in the fix for that),
and even once those were fixed, the fixed-wide camera itself made most acts
imperceptible -- verified numerically, not guessed: a millimeter-scale act
("Your Name on a Grain of Rice", 0.006m amplitude) occupied ~0.14% of a
~4.25m-wide shared-stage camera frame.

The current version gives every act its own subplot pane (small multiples),
each auto-zoomed to its own trajectory's own scale, and resends all three
per-act traces (body, glow, trail) on every frame for every act -- no
permanent-slot bookkeeping left to get wrong. These tests are pure
Figure-structure checks (no hardware, no browser).
"""

import numpy as np
import pytest
from robot_viz import build_multi_dancer_figure


def _make_slot(name: str, n_steps: int, y_offset: float, speed: float = 1.0) -> dict:
    poses = np.zeros((n_steps + 1, 7), dtype=np.float32)
    poses[:, 0] = np.linspace(0, 0.1, n_steps + 1)
    poses[:, 1] = y_offset
    return dict(
        name=name,
        poses=poses,
        energies=np.linspace(0.1, 0.9, n_steps),
        latencies_ms=np.full(n_steps, 5.0),
        pen_up=np.zeros(n_steps, dtype=bool),
        labels=[f"{name}-step{i}" for i in range(n_steps)],
        palette_segments=[("red", 0, n_steps)],
        speed=speed,
    )


@pytest.mark.parametrize("n_slots", [1, 2, 3, 7])
def test_base_figure_has_exactly_three_traces_per_act(n_slots):
    slots = [_make_slot(f"dancer{i}", 3, i * 0.5) for i in range(n_slots)]
    fig, _ = build_multi_dancer_figure(slots)

    assert len(fig.data) == 3 * n_slots


def test_every_frame_touches_every_acts_three_traces():
    """Every frame resends body/glow/trail for every act (active, idle, or frozen) --
    the whole point of dropping the permanent-slot scheme was that every frame's
    trace set is now the same, full set, every time."""
    n_slots = 5
    slots = [_make_slot(f"dancer{i}", 3, i * 0.5) for i in range(n_slots)]
    fig, _ = build_multi_dancer_figure(slots)

    expected_traces = set(range(3 * n_slots))
    for f in fig.frames:
        assert f.traces is not None, f"frame {f.name} has no explicit traces= mapping"
        assert set(f.traces) == expected_traces, f"frame {f.name} didn't touch every act's traces"
        assert len(f.data) == len(f.traces)


def test_each_act_gets_its_own_axis_range_scaled_to_its_own_motion():
    """The whole point of the pane-per-act redesign: a millimeter-scale act's pane
    must NOT share a giant multi-meter axis range with a normal-amplitude act --
    each pane's range should be small enough that its own motion actually fills it."""
    slots = [
        _make_slot("tiny", 3, 0.0),  # default amplitude ~0.1m over 3 steps
        _make_slot("normal", 3, 5.0),  # far away in y, but that must not matter now
    ]
    slots[0]["poses"][:, 0] = np.linspace(0, 0.001, 4)  # shrink to ~1mm motion
    fig, _ = build_multi_dancer_figure(slots)

    tiny_range = fig.layout["scene"].xaxis.range
    normal_range = fig.layout["scene2"].xaxis.range
    tiny_span = tiny_range[1] - tiny_range[0]
    normal_span = normal_range[1] - normal_range[0]
    assert tiny_span < normal_span, "the tiny-motion act's pane should auto-zoom tighter"


def test_frame_audio_aligned_with_frames_and_well_formed():
    n_slots = 7
    slots = [_make_slot(f"dancer{i}", 3, i * 0.5) for i in range(n_slots)]
    fig, frame_audio = build_multi_dancer_figure(slots)

    assert len(frame_audio) == len(fig.frames)
    for entry in frame_audio:
        assert set(entry) == {
            "palette",
            "voice",
            "energy",
            "latency_ms",
            "energy_norm",
            "latency_norm",
            "duration_ms",
        }
        assert 0.0 <= entry["energy_norm"] <= 1.0
        assert 0.0 <= entry["latency_norm"] <= 1.0
        assert entry["duration_ms"] > 0


def test_slower_speed_produces_longer_frame_duration():
    n_steps = 3
    slots = [_make_slot("slow", n_steps, 0.0, speed=0.5), _make_slot("fast", n_steps, 1.0, speed=2.0)]
    _, frame_audio = build_multi_dancer_figure(slots)

    frames_per_slot = n_steps + 1  # poses is [n_steps+1, 7] -> one frame per pose
    slow_durations = [e["duration_ms"] for e in frame_audio[:frames_per_slot]]
    fast_durations = [e["duration_ms"] for e in frame_audio[frames_per_slot : 2 * frames_per_slot]]
    assert min(slow_durations) > max(fast_durations)


def test_curtain_call_frames_are_silent():
    n_slots = 2
    slots = [_make_slot(f"dancer{i}", 3, i * 0.5) for i in range(n_slots)]
    fig, frame_audio = build_multi_dancer_figure(slots, curtain_call_frames=3)

    curtain_entries = frame_audio[-3:]
    assert all(e["voice"] == "silence" and e["palette"] == "curtain" for e in curtain_entries)


def test_every_palette_used_by_the_real_show_has_a_voice_mapping():
    from robot_viz import PALETTE_VOICES

    real_show_palettes = {"red", "cool", "flash", "tangerine", "micro", "rainbow", "letters"}
    assert real_show_palettes <= set(PALETTE_VOICES)
