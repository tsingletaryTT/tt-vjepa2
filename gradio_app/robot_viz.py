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

Body geometry (2026-09-04 upgrade): the 2-link arm used to be drawn as two flat lines
-- "wiggling lines," not a robot. Each limb segment now gets a solid chrome Mesh3d
cylinder underneath the original color-law line, which is kept exactly as-is and now
reads as a glowing spine down the chrome body rather than the only visible geometry.
Joints stay as Scatter3d markers (a real shaded sphere mesh was cut for time -- at this
scale a bright marker dot already reads as a glossy joint). The trail is now a decaying
particle chain instead of one translucent line.

Per-act panes (2026-09-04, same day, second pass): a first version of this upgrade put
every dancer on one shared 3D stage with a single fixed wide camera -- and that camera
had to frame the *whole* stage (every act's own spot) to always keep the current dancer
in view. Measured, not guessed: with 7 acts spaced 0.55m apart, that camera frames a
~4.25m-wide stage, and "Your Name on a Grain of Rice" is a deliberately millimeter-scale
gesture (0.006m amplitude) -- 0.14% of that frame width. Genuinely invisible, not a
rendering bug. Fixed by giving every act its own subplot pane (a small-multiples grid,
via plotly.subplots.make_subplots) with its OWN bbox computed only from its own
trajectory -- each pane is naturally zoomed to its own act's scale for free, and a
slow-motion act just gets a longer per-frame duration (moves.build_show's per-act
`speed` field), independent of every other act's pacing. The shared-stage light
pool/spotlight-cone concept from the first pass doesn't apply to separate panes and was
dropped; the "who's dancing right now" cue is now a highlighted border around the
active pane instead of a 3D spotlight beam. See build_multi_dancer_figure's own
docstring for the per-frame audio metadata this still carries.
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


def step_color(
    step_idx: int, n_steps: int, energy: float, energy_ref: float, latency_ms: float, latency_ref_ms: float
) -> str:
    """tt-toplike's HSV law, ported, hue pinned to red (Kraftwerk red/black/white
    instead of a rainbow sweep): saturation runs white-hot -> pure red as the model's
    per-step "energy" rises, value (brightness) from real measured latency."""
    energy_norm = float(np.clip(energy / max(energy_ref, 1e-6), 0.0, 1.0))
    hue = 0.0
    saturation = float(np.clip(0.15 + 0.85 * energy_norm, 0.0, 1.0))
    value = float(np.clip(0.55 + 0.45 * np.clip(latency_ms / max(latency_ref_ms, 1e-6), 0.0, 1.5) / 1.5, 0.55, 1.0))
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def step_color_palette(
    step_idx: int,
    n_steps: int,
    energy: float,
    energy_ref: float,
    latency_ms: float,
    latency_ref_ms: float,
    palette: str = "red",
) -> str:
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


# --- Chrome body geometry -------------------------------------------------------------
#
# Pure-numpy mesh builders. Kept dependency-free (no trimesh/etc) since each is a
# handful of triangles generated fresh every frame (the arm moves every step).

CHROME_COLOR = "rgb(150,150,155)"
CHROME_LIGHTING = dict(ambient=0.35, diffuse=0.8, specular=0.9, roughness=0.25, fresnel=0.3)
CHROME_LIGHTPOS = dict(x=200, y=300, z=400)


def _ring_basis(axis_dir: np.ndarray):
    """Two unit vectors perpendicular to `axis_dir`, for building a circular
    cross-section around a 3D axis (same trick as two_link_elbow's `perp`)."""
    up = np.array([0.0, 0.0, 1.0])
    ref = up if abs(np.dot(axis_dir, up)) < 0.95 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis_dir, ref)
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(axis_dir, u)
    return u, v


def _cylinder_verts_faces(p0: np.ndarray, p1: np.ndarray, radius: float, n_sides: int, index_offset: int):
    """One tapered-radius-free cylinder from p0 to p1 (capped both ends). Returns
    (verts [2*n_sides+2, 3], faces (i,j,k) each a flat list, already offset by
    `index_offset` so several of these can be concatenated into one Mesh3d)."""
    axis = p1 - p0
    length = np.linalg.norm(axis)
    axis_dir = axis / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
    u, v = _ring_basis(axis_dir)
    thetas = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring_dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1) @ np.stack([u, v])  # [n_sides,3]
    bottom_ring = p0 + radius * ring_dirs
    top_ring = p1 + radius * ring_dirs
    verts = np.concatenate([bottom_ring, top_ring, p0[None, :], p1[None, :]], axis=0)
    bottom_center = index_offset + 2 * n_sides
    top_center = index_offset + 2 * n_sides + 1
    i_idx, j_idx, k_idx = [], [], []
    for s in range(n_sides):
        s2 = (s + 1) % n_sides
        b0, b1 = index_offset + s, index_offset + s2
        t0, t1 = index_offset + n_sides + s, index_offset + n_sides + s2
        # side wall, split into 2 triangles
        i_idx += [b0, b0]
        j_idx += [b1, t1]
        k_idx += [t1, t0]
        # end caps
        i_idx += [bottom_center, top_center]
        j_idx += [b1, t0]
        k_idx += [b0, t1]
    return verts, (i_idx, j_idx, k_idx)


def _limb_mesh(
    base: np.ndarray,
    elbow: np.ndarray,
    gripper: np.ndarray,
    color: str,
    r1: float = 0.035,
    r2: float = 0.022,
    n_sides: int = 8,
    opacity: float = 1.0,
):
    """Both limb segments (base->elbow, elbow->gripper) fused into one Mesh3d trace --
    tapered (r1 thicker at the shoulder, r2 thinner at the wrist), so it reads as one
    continuous chrome body, not two disjoint tubes."""
    import plotly.graph_objects as go

    v1, (i1, j1, k1) = _cylinder_verts_faces(base, elbow, r1, n_sides, 0)
    v2, (i2, j2, k2) = _cylinder_verts_faces(elbow, gripper, r2, n_sides, len(v1))
    verts = np.concatenate([v1, v2], axis=0)
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i1 + i2,
        j=j1 + j2,
        k=k1 + k2,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting=CHROME_LIGHTING,
        lightposition=CHROME_LIGHTPOS,
        showscale=False,
        showlegend=False,
        hoverinfo="skip",
    )


def _empty_mesh():
    import plotly.graph_objects as go

    return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], showlegend=False, visible=False, hoverinfo="skip")


# Per-act audio "voice": which palette maps to which sonification texture client-side
# (disco/wrap_iframe embeds the actual synthesis -- this table is just the mapping
# name so app.py and robot_viz.py agree on the vocabulary; see app.py's AUTO_LOOP_SCRIPT
# for the Web Audio implementation of each voice).
PALETTE_VOICES = {
    "tangerine": "ratchet",
    "cool": "soft",
    "flash": "stab",
    "micro": "ticks",
    "rainbow": "arpeggio",
    "red": "motorik",
    "letters": "chime",
}


def build_multi_dancer_figure(slots: list, l1: float = 0.28, l2: float = 0.28, curtain_call_frames: int = 3):
    """Each entry in `slots` is one independently-rolled-out 'dancer', now given its
    OWN subplot pane in a small-multiples grid (plotly.subplots.make_subplots) rather
    than a shared 3D stage. A first version put every dancer on one shared stage with
    a single fixed wide camera that had to frame the *whole* stage to always keep the
    current dancer in view -- measured, not guessed: with 7 acts spaced 0.55m apart
    that camera framed a ~4.25m-wide stage, and "Your Name on a Grain of Rice" is a
    deliberately millimeter-scale gesture (0.006m amplitude) -- 0.14% of that frame
    width, genuinely invisible rather than a rendering bug. Each pane here gets its OWN
    bbox computed only from its own trajectory, so it's naturally zoomed to its own
    act's scale for free; a slow-motion act (moves.build_show's per-act `speed` field)
    just gets a longer per-frame duration, independent of every other act.

    All acts are visible at once (grid), but only one is animating at any moment: the
    currently-dancing pane gets a highlighted border (a paper-space rect shape sized to
    that pane's own subplot domain) plus the small corner HUD; not-yet-started panes
    hold their first pose, dim; finished panes hold their final pose, dim. Every
    frame resends all three per-act traces (chrome body, glow spine, particle trail)
    for every act, not just the active one -- simpler and far less bug-prone than the
    permanent-trace-slot bookkeeping the single-shared-stage version needed (twice:
    once for the original "only 2 dancers ever show up" trace-index bug, and this
    version fixed a second, separate initial-frame indexing bug in that same scheme).
    Ends on a few 'curtain call' frames with every act frozen and no highlight.

    Returns (fig, frame_audio): frame_audio is a list aligned 1:1 with fig.frames, each
    entry {"palette","voice","energy","latency_ms","energy_norm","latency_norm",
    "duration_ms"} for the client-side Web Audio sonification and per-frame pacing
    (see app.py) -- "voice" is looked up from PALETTE_VOICES.

    Each slot dict needs: name, poses [T+1,7], energies [T], latencies_ms [T],
    pen_up [T] bool, labels [T] str, palette_segments [(palette,start,end), ...] in
    LOCAL (this slot's own) step-index space, speed (float, playback-speed
    multiplier). `zoom`/`camera_bias` are accepted but unused -- kept so callers don't
    need updating."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_slots = len(slots)
    cols = int(np.ceil(np.sqrt(n_slots)))
    rows = int(np.ceil(n_slots / cols))

    for slot in slots:
        positions = slot["poses"][:, :3]
        direction = np.array([-1.0, 0.0, -0.15])
        direction = direction / np.linalg.norm(direction)
        slot["base"] = positions[0] + direction * (l1 + l2) * 0.6
        slot["energy_ref"] = float(np.max(slot["energies"])) if len(slot["energies"]) else 1.0
        slot["latency_ref"] = float(np.median(slot["latencies_ms"])) if len(slot["latencies_ms"]) else 1.0
        # Per-act bbox: computed ONLY from this act's own trajectory + base -- this is
        # what fixes the zoom problem; a millimeter-scale act naturally gets a
        # millimeter-scale (i.e. genuinely zoomed-in) axis range.
        bbox_min = np.minimum(positions.min(axis=0), slot["base"]) - 0.05
        bbox_max = np.maximum(positions.max(axis=0), slot["base"]) + 0.05
        half_range = max((bbox_max - bbox_min).max() / 2, 0.05)
        center = (bbox_min + bbox_max) / 2
        slot["axis_ranges"] = {
            "x": [center[0] - half_range, center[0] + half_range],
            "y": [center[1] - half_range, center[1] + half_range],
            "z": [max(center[2] - half_range, 0.0), center[2] + half_range],
        }

    def palette_for(slot, local_action_idx):
        for palette, lo, hi in slot["palette_segments"]:
            if lo <= local_action_idx < hi:
                return palette
        return slot["palette_segments"][-1][0]

    grid_axis = dict(
        gridcolor="#333333", zerolinecolor="#555555", color="#dddddd", showbackground=True, backgroundcolor="#000000"
    )
    PANE_CAMERA = dict(center=dict(x=0, y=0, z=0), eye=dict(x=1.8, y=1.8, z=1.3), up=dict(x=0, y=0, z=1))
    TRAIL_LEN = 12  # ring-buffer length for the particle trail

    specs = [[None] * cols for _ in range(rows)]
    subplot_titles = [""] * (rows * cols)
    scene_names = [None] * n_slots
    scene_counter = 0
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        if idx < n_slots:
            specs[r][c] = {"type": "scene"}
            subplot_titles[idx] = slots[idx]["name"]
            scene_counter += 1
            scene_names[idx] = "scene" if scene_counter == 1 else f"scene{scene_counter}"
    grid_fig = make_subplots(
        rows=rows, cols=cols, specs=specs, subplot_titles=subplot_titles, horizontal_spacing=0.03, vertical_spacing=0.08
    )
    base_annotations = list(grid_fig.layout.annotations)  # the subplot-title annotations
    scene_domains = {i: grid_fig.layout[scene_names[i]].domain for i in range(n_slots)}

    for i, slot in enumerate(slots):
        grid_fig.layout[scene_names[i]].update(
            xaxis=dict(range=slot["axis_ranges"]["x"], **grid_axis, showticklabels=False, title=""),
            yaxis=dict(range=slot["axis_ranges"]["y"], **grid_axis, showticklabels=False, title=""),
            zaxis=dict(range=slot["axis_ranges"]["z"], **grid_axis, showticklabels=False, title=""),
            aspectmode="cube",
            bgcolor="#000000",
            camera=PANE_CAMERA,
        )

    def render_pose(slot, pose_idx, energy_val, latency_val, palette, opacity):
        pos = slot["poses"][pose_idx, :3]
        elbow = two_link_elbow(slot["base"], pos, l1, l2)
        gripper_open = 1.0 - slot["poses"][pose_idx, 6]
        color = step_color_palette(
            0,
            1,
            energy=energy_val,
            energy_ref=slot["energy_ref"],
            latency_ms=latency_val,
            latency_ref_ms=slot["latency_ref"],
            palette=palette,
        )
        body = _limb_mesh(slot["base"], elbow, pos, CHROME_COLOR, opacity=opacity)
        glow = go.Scatter3d(
            x=[slot["base"][0], elbow[0], pos[0]],
            y=[slot["base"][1], elbow[1], pos[1]],
            z=[slot["base"][2], elbow[2], pos[2]],
            mode="lines+markers",
            line=dict(color=color, width=6),
            marker=dict(size=[5, 6, 10 + 8 * gripper_open], color=color),
            opacity=opacity,
            showlegend=False,
        )
        return body, glow, color

    def idle_render(i, slot):
        """Not yet this act's turn: held at its first pose, dim."""
        palette = slot["palette_segments"][0][0]
        body, glow, _ = render_pose(slot, 0, 0.0, slot["latency_ref"], palette, opacity=0.35)
        body.update(scene=scene_names[i])
        glow.update(scene=scene_names[i])
        return body, glow, go.Scatter3d(x=[], y=[], z=[], mode="markers", scene=scene_names[i], showlegend=False)

    def frozen_render(i, slot):
        """Finished: held at its final pose, dim (brighter than idle -- it did dance)."""
        palette = slot["palette_segments"][-1][0]
        n_local = len(slot["poses"])
        body, glow, _ = render_pose(slot, n_local - 1, 0.0, slot["latency_ref"], palette, opacity=0.5)
        body.update(scene=scene_names[i])
        glow.update(scene=scene_names[i])
        return body, glow, go.Scatter3d(x=[], y=[], z=[], mode="markers", scene=scene_names[i], showlegend=False)

    def BODY_IDX(i):
        return 3 * i

    def GLOW_IDX(i):
        return 3 * i + 1

    def TRAIL_IDX(i):
        return 3 * i + 2

    def highlight_shape(i, color):
        dom = scene_domains[i]
        pad_x, pad_y = 0.01, 0.02
        return dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=dom.x[0] - pad_x,
            x1=dom.x[1] + pad_x,
            y0=dom.y[0] - pad_y,
            y1=dom.y[1] + pad_y,
            line=dict(color=color, width=3),
            fillcolor="rgba(0,0,0,0)",
        )

    frames = []
    frame_audio = []
    total_steps = sum(len(s["energies"]) for s in slots)
    global_step = 0
    for slot_i, slot in enumerate(slots):
        poses = slot["poses"]
        n_local = len(poses)
        duration_ms = max(int(220 / max(slot["speed"], 0.05)), 30)
        trail_pts = []  # ring buffer for the particle trail
        for i in range(n_local):
            pos = poses[i, :3]
            if i > 0 and not slot["pen_up"][i - 1]:
                trail_pts.append(pos)
                trail_pts = trail_pts[-TRAIL_LEN:]

            local_action_idx = max(i - 1, 0)
            palette = palette_for(slot, local_action_idx)
            energy_val = slot["energies"][i - 1] if i > 0 else 0.0
            latency_val = slot["latencies_ms"][i - 1] if i > 0 else slot["latency_ref"]
            body, glow, color = render_pose(slot, i, energy_val, latency_val, palette, opacity=1.0)
            body.update(scene=scene_names[slot_i])
            glow.update(scene=scene_names[slot_i])

            n_trail = len(trail_pts)
            if n_trail:
                trail_arr = np.stack(trail_pts)
                fade = np.linspace(0.15, 0.9, n_trail)  # oldest point dimmest
                trail = go.Scatter3d(
                    x=trail_arr[:, 0],
                    y=trail_arr[:, 1],
                    z=trail_arr[:, 2],
                    mode="markers",
                    marker=dict(size=(4 + 6 * fade).tolist(), color=color, opacity=0.7),
                    scene=scene_names[slot_i],
                    showlegend=False,
                )
            else:
                trail = go.Scatter3d(x=[], y=[], z=[], mode="markers", scene=scene_names[slot_i], showlegend=False)

            frame_traces = [BODY_IDX(slot_i), GLOW_IDX(slot_i), TRAIL_IDX(slot_i)]
            frame_data = [body, glow, trail]
            for j, other in enumerate(slots):
                if j == slot_i:
                    continue
                b, g, t = frozen_render(j, other) if j < slot_i else idle_render(j, other)
                frame_traces += [BODY_IDX(j), GLOW_IDX(j), TRAIL_IDX(j)]
                frame_data += [b, g, t]

            label = slot["labels"][local_action_idx] if local_action_idx < len(slot["labels"]) else slot["name"]
            hud_text = (
                f"{slot['name']}<br>"
                f"STEP {global_step + 1}/{total_steps} &nbsp;·&nbsp; {label}<br>"
                f"ENERGY {energy_val:.3f} &nbsp;·&nbsp; {latency_val:.1f} ms"
            )
            hud_annotation = dict(
                text=hud_text,
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.01,
                showarrow=False,
                align="left",
                font=dict(family="'Orbitron', monospace", size=11, color=color),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=color,
                borderwidth=1,
            )
            frame_layout = go.Layout(
                shapes=[highlight_shape(slot_i, color)],
                annotations=base_annotations + [hud_annotation],
            )
            frames.append(go.Frame(data=frame_data, traces=frame_traces, name=str(len(frames)), layout=frame_layout))
            energy_norm = float(np.clip(energy_val / max(slot["energy_ref"], 1e-6), 0.0, 1.0))
            latency_norm = float(np.clip(latency_val / max(slot["latency_ref"], 1e-6), 0.0, 1.5) / 1.5)
            frame_audio.append(
                dict(
                    palette=palette,
                    voice=PALETTE_VOICES.get(palette, "motorik"),
                    energy=energy_val,
                    latency_ms=latency_val,
                    energy_norm=energy_norm,
                    latency_norm=latency_norm,
                    duration_ms=duration_ms,
                )
            )
            if i > 0:
                global_step += 1

    # Curtain call: every act frozen at its final pose, no highlight.
    for _ in range(curtain_call_frames):
        frame_traces, frame_data = [], []
        for j, other in enumerate(slots):
            b, g, t = frozen_render(j, other)
            frame_traces += [BODY_IDX(j), GLOW_IDX(j), TRAIL_IDX(j)]
            frame_data += [b, g, t]
        frames.append(
            go.Frame(
                data=frame_data,
                traces=frame_traces,
                name=str(len(frames)),
                layout=go.Layout(
                    shapes=[],
                    annotations=base_annotations
                    + [
                        dict(
                            text="CURTAIN CALL",
                            xref="paper",
                            yref="paper",
                            x=0.01,
                            y=0.01,
                            showarrow=False,
                            align="left",
                            font=dict(family="'Orbitron', monospace", size=14, color="#ff2b2b"),
                            bgcolor="rgba(0,0,0,0.6)",
                            bordercolor="#ff2b2b",
                            borderwidth=1,
                        )
                    ],
                ),
            )
        )
        frame_audio.append(
            dict(
                palette="curtain",
                voice="silence",
                energy=0.0,
                latency_ms=0.0,
                energy_norm=0.0,
                latency_norm=0.0,
                duration_ms=220,
            )
        )

    n_frames = len(frames)
    initial_data = [None] * (3 * n_slots)
    for i, slot in enumerate(slots):
        b, g, t = idle_render(i, slot)
        initial_data[BODY_IDX(i)] = b
        initial_data[GLOW_IDX(i)] = g
        initial_data[TRAIL_IDX(i)] = t
    grid_fig.add_traces(initial_data)
    grid_fig.frames = frames
    grid_fig.update_layout(
        template="plotly_dark",
        font=dict(family="'Orbitron', 'Michroma', monospace", color="#ff2b2b"),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(320 * rows, 480),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                bgcolor="#1a0000",
                font=dict(color="#ff2b2b"),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 220, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                font=dict(color="#ff2b2b"),
                bgcolor="#000000",
                activebgcolor="#ff2b2b",
                currentvalue=dict(prefix="STEP: ", font=dict(color="#ff2b2b")),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(i),
                    )
                    for i in range(n_frames)
                ],
            )
        ],
    )
    return grid_fig, frame_audio


def build_dance_figure(
    poses: np.ndarray,
    energies: np.ndarray,
    latencies_ms: np.ndarray,
    base: np.ndarray | None = None,
    l1: float = 0.28,
    l2: float = 0.28,
):
    """poses: [N,7] end-effector poses (from integrate_pose). energies: [N-1] per-step
    model signal (predicted-embedding delta between consecutive imagined frames).
    latencies_ms: [N-1] real measured predictor forward-pass time for that step.
    Returns a plotly.graph_objects.Figure with one animation frame per step.

    `base` defaults to a point placed relative to the trajectory's own starting
    position (not a fixed world coordinate) -- real DROID/Franka poses live wherever
    that robot's base frame puts them (here, around x=0.6m), and a fixed guess like
    the origin can end up farther from the trajectory than l1+l2, permanently
    maxing out the 2-link reach into a straight, seemingly frozen rod.

    Kept as the plain 2-line stick-arm rendering deliberately -- this is the
    experimentation/testing tab (arbitrary typed moves, no palette/voice metadata
    attached), not the flagship Show; see build_multi_dancer_figure for the chrome
    body / particle trail / stage lighting upgrade."""
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
        trail_x.append(pos[0])
        trail_y.append(pos[1])
        trail_z.append(pos[2])
        elbow = two_link_elbow(base, pos, l1, l2)
        gripper_open = 1.0 - pose[6]
        color = step_color(
            i,
            n_steps,
            energy=energies[i - 1] if i > 0 else 0.0,
            energy_ref=energy_ref,
            latency_ms=latencies_ms[i - 1] if i > 0 else latency_ref,
            latency_ref_ms=latency_ref,
        )
        marker_size = 10 + 14 * gripper_open

        arm_trace = go.Scatter3d(
            x=[base[0], elbow[0], pos[0]],
            y=[base[1], elbow[1], pos[1]],
            z=[base[2], elbow[2], pos[2]],
            mode="lines+markers",
            line=dict(color=color, width=10),
            marker=dict(size=[6, 8, marker_size], color=color),
            showlegend=False,
        )
        trail_trace = go.Scatter3d(
            x=trail_x,
            y=trail_y,
            z=trail_z,
            mode="lines",
            line=dict(color="rgba(255,60,60,0.35)", width=3),
            showlegend=False,
        )
        frames.append(go.Frame(data=[arm_trace, trail_trace], name=str(i)))

    grid_axis = dict(
        gridcolor="#3a0808", zerolinecolor="#5a0d0d", color="#ff2b2b", showbackground=True, backgroundcolor="#000000"
    )
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
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                bgcolor="#1a0000",
                font=dict(color="#ff2b2b"),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 220, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                font=dict(color="#ff2b2b"),
                bgcolor="#000000",
                activebgcolor="#ff2b2b",
                currentvalue=dict(prefix="STEP: ", font=dict(color="#ff2b2b")),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(i),
                    )
                    for i in range(n_steps)
                ],
            )
        ],
    )
    return fig
