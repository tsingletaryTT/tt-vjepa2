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


def step_color_palette(step_idx: int, n_steps: int, energy: float, energy_ref: float,
                        latency_ms: float, latency_ref_ms: float, palette: str = "red") -> str:
    """Same tt-toplike HSV law (real signals drive hue/saturation/value), but with a
    per-act palette instead of always red -- 'red everywhere' reads as one loud tone,
    not six distinct acts. Each palette still ties at least one channel to a real
    per-step signal (energy or latency); only 'flash' is a deliberate, momentary
    exception (the one dramatic beat that's meant to read as maxed-out, not measured)."""
    energy_norm = float(np.clip(energy / max(energy_ref, 1e-6), 0.0, 1.0))
    latency_norm = float(np.clip(latency_ms / max(latency_ref_ms, 1e-6), 0.0, 1.5) / 1.5)
    if palette == "tangerine":
        hue, saturation = 0.07, float(np.clip(0.2 + 0.8 * energy_norm, 0.0, 1.0))
        value = float(np.clip(0.55 + 0.45 * latency_norm, 0.55, 1.0))
    elif palette == "cool":
        hue, saturation = 0.56, 0.55
        value = float(np.clip(0.22 + 0.55 * energy_norm, 0.2, 0.85))
    elif palette == "flash":
        hue, saturation, value = 0.0, 1.0, 1.0
    elif palette == "micro":
        hue, saturation = 0.5, 0.08
        value = float(np.clip(0.45 + 0.3 * energy_norm, 0.4, 0.8))
    elif palette == "rainbow":
        hue, saturation = (step_idx / max(n_steps - 1, 1)) % 1.0, 1.0
        value = float(np.clip(0.55 + 0.45 * latency_norm, 0.55, 1.0))
    else:  # "red" (default / Instant Kraftwerk / Poppin and Lockin)
        hue, saturation = 0.0, float(np.clip(0.15 + 0.85 * energy_norm, 0.0, 1.0))
        value = float(np.clip(0.55 + 0.45 * latency_norm, 0.55, 1.0))
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def build_multi_dancer_figure(slots: list, l1: float = 0.28, l2: float = 0.28,
                               curtain_call_frames: int = 3):
    """Each entry in `slots` is one independently-rolled-out 'dancer' at its own fixed
    spot on a shared stage (see app.py's run_show for how slots are built -- each
    slot's poses already include that spot's world-space offset). As the show moves
    from slot to slot, earlier dancers freeze in their final pose at their own spot
    instead of one dancer relocating through every act; the camera recenters and
    reframes on whichever slot is currently animating (same tt-toplike-inspired
    per-act palette + live HUD as before, just per-slot now). Ends on a few
    wide-establishing 'curtain call' frames showing every dancer at once before the
    show loops back to the top.

    Each slot dict needs: name, poses [T+1,7], energies [T], latencies_ms [T],
    pen_up [T] bool, labels [T] str, palette_segments [(palette,start,end), ...] in
    LOCAL (this slot's own) step-index space, zoom, camera_bias."""
    import plotly.graph_objects as go

    for slot in slots:
        positions = slot["poses"][:, :3]
        direction = np.array([-1.0, 0.0, -0.15])
        direction = direction / np.linalg.norm(direction)
        slot["base"] = positions[0] + direction * (l1 + l2) * 0.6
        slot["energy_ref"] = float(np.max(slot["energies"])) if len(slot["energies"]) else 1.0
        slot["latency_ref"] = float(np.median(slot["latencies_ms"])) if len(slot["latencies_ms"]) else 1.0

    def palette_for(slot, local_action_idx):
        for palette, lo, hi in slot["palette_segments"]:
            if lo <= local_action_idx < hi:
                return palette
        return slot["palette_segments"][-1][0]

    # One global stage: axis ranges span every slot's full trajectory + base, so the
    # whole lineup is always technically in frame -- only the camera's focus and zoom
    # change per slot, which is what makes "leave a dancer, refocus on the next" (and
    # the final curtain call) legible instead of the grid itself jumping around.
    all_pts = np.concatenate(
        [s["poses"][:, :3] for s in slots] + [s["base"][None, :] for s in slots], axis=0
    )
    margin = 0.2
    bbox_min = all_pts.min(axis=0) - margin
    bbox_max = all_pts.max(axis=0) + margin
    half_range = max((bbox_max - bbox_min).max() / 2, 0.2)
    center = (bbox_min + bbox_max) / 2
    axis_ranges = {
        "x": [center[0] - half_range, center[0] + half_range],
        "y": [center[1] - half_range, center[1] + half_range],
        "z": [max(center[2] - half_range, 0.0), center[2] + half_range],
    }

    # A per-act camera fully recentered + tightly zoomed on just that act (the
    # single-dancer version's approach) hides every other dancer entirely once the
    # stage is wide enough to hold several of them -- found this by actually looking
    # at a render, not assuming it. Instead: blend only partway toward each act's own
    # position (BLEND_TOWARD_ACT), and require a minimum eye distance that grows with
    # how wide the whole staged lineup is, so "focus on the current dancer" reads as
    # emphasis on an otherwise-visible stage, not a tight solo close-up.
    BLEND_TOWARD_ACT = 0.4
    stage_half_width_norm = float(np.linalg.norm((bbox_max - bbox_min) / 2) / half_range)

    def camera_for(seg_center, zoom, camera_bias):
        seg_norm_center = (seg_center - center) / half_range
        norm_center = seg_norm_center * BLEND_TOWARD_ACT  # partial recenter, not full
        offset = np.array(camera_bias) * (zoom / 1.5)
        min_dist = max(0.75, stage_half_width_norm * 0.8)  # keep the whole lineup in frame
        dist = np.linalg.norm(offset)
        if dist < min_dist:
            offset = offset / (dist + 1e-9) * min_dist
        eye = norm_center + offset
        return dict(center=dict(x=float(norm_center[0]), y=float(norm_center[1]), z=float(norm_center[2])),
                    eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                    up=dict(x=0, y=0, z=1))

    def frozen_trace(slot):
        """A completed dancer's final pose, held static -- same shape as a live arm
        trace so it composites identically, just no longer changing frame to frame."""
        pos = slot["poses"][-1, :3]
        elbow = two_link_elbow(slot["base"], pos, l1, l2)
        color = step_color_palette(0, 1, energy=0, energy_ref=1, latency_ms=0, latency_ref_ms=1,
                                    palette=slot["palette_segments"][-1][0])
        return go.Scatter3d(
            x=[slot["base"][0], elbow[0], pos[0]], y=[slot["base"][1], elbow[1], pos[1]],
            z=[slot["base"][2], elbow[2], pos[2]], mode="lines+markers",
            line=dict(color=color, width=6), marker=dict(size=[4, 5, 8], color=color),
            opacity=0.55, showlegend=False,
        )

    grid_axis = dict(gridcolor="#333333", zerolinecolor="#555555", color="#dddddd",
                      showbackground=True, backgroundcolor="#000000")

    frames = []
    frozen = []  # completed slots' static traces, carried into every subsequent frame
    total_steps = sum(len(s["energies"]) for s in slots)
    global_step = 0
    for slot_i, slot in enumerate(slots):
        poses = slot["poses"]
        n_local = len(poses)
        trail_x, trail_y, trail_z = [poses[0, 0]], [poses[0, 1]], [poses[0, 2]]
        for i in range(n_local):
            pos = poses[i, :3]
            if i > 0:
                if slot["pen_up"][i - 1]:
                    trail_x.append(None); trail_y.append(None); trail_z.append(None)
                trail_x.append(pos[0]); trail_y.append(pos[1]); trail_z.append(pos[2])

            local_action_idx = max(i - 1, 0)
            palette = palette_for(slot, local_action_idx)
            elbow = two_link_elbow(slot["base"], pos, l1, l2)
            gripper_open = 1.0 - poses[i, 6]
            color = step_color_palette(
                i, n_local,
                energy=slot["energies"][i - 1] if i > 0 else 0.0, energy_ref=slot["energy_ref"],
                latency_ms=slot["latencies_ms"][i - 1] if i > 0 else slot["latency_ref"],
                latency_ref_ms=slot["latency_ref"], palette=palette,
            )
            marker_size = 10 + 14 * gripper_open

            arm_trace = go.Scatter3d(
                x=[slot["base"][0], elbow[0], pos[0]], y=[slot["base"][1], elbow[1], pos[1]],
                z=[slot["base"][2], elbow[2], pos[2]], mode="lines+markers",
                line=dict(color=color, width=10), marker=dict(size=[6, 8, marker_size], color=color),
                showlegend=False,
            )
            trail_trace = go.Scatter3d(
                x=list(trail_x), y=list(trail_y), z=list(trail_z), mode="lines",
                line=dict(color="rgba(200,200,200,0.3)", width=3), showlegend=False,
            )
            label = slot["labels"][local_action_idx] if local_action_idx < len(slot["labels"]) else slot["name"]
            energy_val = slot["energies"][i - 1] if i > 0 else 0.0
            latency_val = slot["latencies_ms"][i - 1] if i > 0 else slot["latency_ref"]
            hud_text = (
                f"{slot['name']}<br>"
                f"STEP {global_step + 1}/{total_steps} &nbsp;·&nbsp; {label}<br>"
                f"ENERGY {energy_val:.3f} &nbsp;·&nbsp; {latency_val:.1f} ms"
            )
            frame_layout = go.Layout(
                scene_camera=camera_for(poses[:, :3].mean(axis=0), slot["zoom"], slot["camera_bias"]),
                annotations=[dict(
                    text=hud_text, xref="paper", yref="paper", x=0.02, y=0.98,
                    showarrow=False, align="left", font=dict(family="'Orbitron', monospace", size=13, color=color),
                    bgcolor="rgba(0,0,0,0.55)", bordercolor=color, borderwidth=1,
                )],
            )
            frames.append(go.Frame(data=frozen + [arm_trace, trail_trace],
                                    name=str(len(frames)), layout=frame_layout))
            if i > 0:
                global_step += 1
        frozen = frozen + [frozen_trace(slot)]

    # Curtain call: a wide shot holding every dancer's final pose at once.
    stage_center = np.stack([s["poses"][-1, :3] for s in slots]).mean(axis=0)
    curtain_camera = camera_for(stage_center, zoom=1.5 * max(1.0, len(slots) / 3), camera_bias=(1.6, 1.6, 1.2))
    for _ in range(curtain_call_frames):
        frames.append(go.Frame(
            data=frozen, name=str(len(frames)),
            layout=go.Layout(scene_camera=curtain_camera, annotations=[dict(
                text="CURTAIN CALL", xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False, align="left", font=dict(family="'Orbitron', monospace", size=16, color="#ff2b2b"),
                bgcolor="rgba(0,0,0,0.55)", bordercolor="#ff2b2b", borderwidth=1,
            )]),
        ))

    n_frames = len(frames)
    fig = go.Figure(data=frames[0].data, layout=frames[0].layout, frames=frames)
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
        height=600,
        updatemenus=[dict(
            type="buttons", showactive=False, bgcolor="#1a0000", font=dict(color="#ff2b2b"),
            buttons=[dict(label="▶ DIE VORFÜHRUNG // START THE SHOW", method="animate",
                          args=[None, {"frame": {"duration": 220, "redraw": True},
                                       "transition": {"duration": 0}, "fromcurrent": True}])],
        )],
        sliders=[dict(
            font=dict(color="#ff2b2b"), bgcolor="#000000", activebgcolor="#ff2b2b",
            currentvalue=dict(prefix="STEP: ", font=dict(color="#ff2b2b")),
            steps=[dict(method="animate", args=[[str(i)], {"mode": "immediate",
                        "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                        label=str(i)) for i in range(n_frames)],
        )],
    )
    return fig


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
            currentvalue=dict(prefix="STEP: ", font=dict(color="#ff2b2b")),
            steps=[dict(method="animate", args=[[str(i)], {"mode": "immediate",
                        "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                        label=str(i)) for i in range(n_steps)],
        )],
    )
    return fig
