# SPDX-License-Identifier: MIT
"""Forward kinematics + a Plotly 3D figure for the 'dancing' end-effector demo.

V-JEPA2-AC doesn't render pixels or move joints -- it predicts embeddings and scores
actions. The only physically real thing we can draw is the end-effector's Cartesian
pose (position + orientation + gripper openness), integrated from a sequence of 7-DoF
actions (dx,dy,dz,droll,dpitch,dyaw,dgripper) exactly the way Meta's own
`compute_new_pose` (reference/notebooks/utils/mpc_utils.py) does it. The 2-link arm
drawn reaching for that end-effector is a stylized illustration, not a real robot's
joint kinematics -- we only ever have the wrist pose, never shoulder/elbow angles.

Color law is tt-toplike's rule (see its README's "How colors are chosen"), constrained
to a Kraftwerk "Die Mensch-Maschine" red/black/white palette instead of a full rainbow:
every color still comes from hsv_to_rgb(hue, saturation, value), still driven by real
signals, but hue is pinned at 0 (red) rather than swept. Saturation runs from white-hot
down to pure red as the model's own per-step "energy" rises (mirrors tt-toplike's
temp_to_hue: cool/pale when calm, saturated red when the signal is high); value
(brightness) is real measured per-step forward-pass latency, so a brighter step is
genuinely doing more device work, not a canned effect.
"""

import colorsys

import numpy as np
from scipy.spatial.transform import Rotation


def integrate_pose(start_pose: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """start_pose: [7] (x,y,z,roll,pitch,yaw,gripper). actions: [N, 7] deltas, same
    convention as Meta's `compute_new_pose`. Returns [N+1, 7]: start_pose followed by
    the pose after each action."""
    poses = [start_pose.copy()]
    pose = start_pose.copy()
    for action in actions:
        xyz = pose[:3] + action[:3]
        r_pose = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
        r_delta = Rotation.from_euler("xyz", action[3:6], degrees=False).as_matrix()
        rpy = Rotation.from_matrix(r_delta @ r_pose).as_euler("xyz", degrees=False)
        gripper = np.clip(pose[6:7] + action[6:7], 0.0, 1.0)
        pose = np.concatenate([xyz, rpy, gripper])
        poses.append(pose.copy())
    return np.stack(poses)


def two_link_elbow(base: np.ndarray, target: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """Stylized 2-link IK: elbow position for an arm reaching from `base` to `target`
    with segment lengths l1 (base->elbow), l2 (elbow->target). Not a real robot's
    kinematics (no shoulder/elbow angles exist in our data, only the wrist pose) --
    purely a "something is reaching for the end-effector" visual."""
    d = target - base
    r = np.clip(np.linalg.norm(d), 1e-3, l1 + l2 - 1e-6)
    u = d / r
    a = np.clip((r**2 + l1**2 - l2**2) / (2 * r), -l1, l1)
    h = np.sqrt(max(l1**2 - a**2, 0.0))
    up = np.array([0.0, 0.0, 1.0])
    ref = up if abs(np.dot(u, up)) < 0.95 else np.array([1.0, 0.0, 0.0])
    perp = np.cross(u, ref)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    return base + a * u + h * perp


def step_color(step_idx: int, n_steps: int, energy: float, energy_ref: float,
               latency_ms: float, latency_ref_ms: float) -> str:
    """tt-toplike's HSV law, ported, hue pinned to red (Kraftwerk red/black/white
    instead of a rainbow sweep): saturation runs white-hot -> pure red as the model's
    per-step "energy" rises, value (brightness) from real measured latency."""
    energy_norm = float(np.clip(energy / max(energy_ref, 1e-6), 0.0, 1.0))
    hue = 0.0
    saturation = float(np.clip(0.15 + 0.85 * energy_norm, 0.0, 1.0))
    value = float(np.clip(0.55 + 0.45 * np.clip(latency_ms / max(latency_ref_ms, 1e-6), 0.0, 1.5) / 1.5, 0.55, 1.0))
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def build_dance_figure(poses: np.ndarray, energies: np.ndarray, latencies_ms: np.ndarray,
                        base: np.ndarray | None = None, l1: float = 0.28, l2: float = 0.28):
    """poses: [N,7] end-effector poses (from integrate_pose). energies: [N-1] per-step
    model signal (predicted-embedding delta between consecutive imagined frames).
    latencies_ms: [N-1] real measured predictor forward-pass time for that step.
    Returns a plotly.graph_objects.Figure with one animation frame per step.

    `base` defaults to a point placed relative to the trajectory's own starting
    position (not a fixed world coordinate) -- real DROID/Franka poses live wherever
    that robot's base frame puts them (here, around x=0.6m), and a fixed guess like
    the origin can end up farther from the trajectory than l1+l2, permanently
    maxing out the 2-link reach into a straight, seemingly frozen rod."""
    import plotly.graph_objects as go

    n_steps = len(poses)
    energy_ref = float(np.max(energies)) if len(energies) else 1.0
    latency_ref = float(np.median(latencies_ms)) if len(latencies_ms) else 1.0

    positions = poses[:, :3]
    if base is None:
        direction = np.array([-1.0, 0.0, -0.15])
        direction = direction / np.linalg.norm(direction)
        base = positions[0] + direction * (l1 + l2) * 0.6  # well within reach, leaves slack

    margin = 0.15
    bbox_min = np.minimum(positions.min(axis=0), base) - margin
    bbox_max = np.maximum(positions.max(axis=0), base) + margin
    half_range = max((bbox_max - bbox_min).max() / 2, 0.2)
    center = (bbox_min + bbox_max) / 2
    axis_ranges = {
        "x": [center[0] - half_range, center[0] + half_range],
        "y": [center[1] - half_range, center[1] + half_range],
        "z": [max(center[2] - half_range, 0.0), center[2] + half_range],
    }

    trail_x, trail_y, trail_z = [], [], []
    frames = []
    for i, pose in enumerate(poses):
        pos = pose[:3]
        trail_x.append(pos[0]); trail_y.append(pos[1]); trail_z.append(pos[2])
        elbow = two_link_elbow(base, pos, l1, l2)
        gripper_open = 1.0 - pose[6]
        color = step_color(
            i, n_steps,
            energy=energies[i - 1] if i > 0 else 0.0,
            energy_ref=energy_ref,
            latency_ms=latencies_ms[i - 1] if i > 0 else latency_ref,
            latency_ref_ms=latency_ref,
        )
        marker_size = 10 + 14 * gripper_open

        arm_trace = go.Scatter3d(
            x=[base[0], elbow[0], pos[0]], y=[base[1], elbow[1], pos[1]], z=[base[2], elbow[2], pos[2]],
            mode="lines+markers", line=dict(color=color, width=10),
            marker=dict(size=[6, 8, marker_size], color=color), showlegend=False,
        )
        trail_trace = go.Scatter3d(
            x=trail_x, y=trail_y, z=trail_z, mode="lines",
            line=dict(color="rgba(255,60,60,0.35)", width=3), showlegend=False,
        )
        frames.append(go.Frame(data=[arm_trace, trail_trace], name=str(i)))

    grid_axis = dict(gridcolor="#3a0808", zerolinecolor="#5a0d0d", color="#ff2b2b",
                      showbackground=True, backgroundcolor="#000000")
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="'Orbitron', 'Michroma', monospace", color="#ff2b2b"),
        scene=dict(
            xaxis=dict(range=axis_ranges["x"], title="x", **grid_axis),
            yaxis=dict(range=axis_ranges["y"], title="y", **grid_axis),
            zaxis=dict(range=axis_ranges["z"], title="z", **grid_axis),
            aspectmode="cube",
            bgcolor="#000000",
        ),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        # No easing/tween between frames -- a rigid, quantized snap from pose to pose
        # (transition duration=0) reads as mechanical gait, not a smooth glide.
        updatemenus=[dict(
            type="buttons", showactive=False, bgcolor="#1a0000", font=dict(color="#ff2b2b"),
            buttons=[dict(label="▶ FUNKTION: TANZEN", method="animate",
                          args=[None, {"frame": {"duration": 220, "redraw": True},
                                       "transition": {"duration": 0}, "fromcurrent": True}])],
        )],
        sliders=[dict(
            font=dict(color="#ff2b2b"), bgcolor="#000000", activebgcolor="#ff2b2b",
            currentvalue=dict(prefix="SCHRITT / STEP: ", font=dict(color="#ff2b2b")),
            steps=[dict(method="animate", args=[[str(i)], {"mode": "immediate",
                        "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                        label=str(i)) for i in range(n_steps)],
        )],
    )
    return fig
