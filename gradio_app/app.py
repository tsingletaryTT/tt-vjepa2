# SPDX-License-Identifier: MIT
"""V-JEPA2-AC demo: what the world model actually does (score/predict in embedding
space, not render pixels), shown two ways.

Tab 1, "Grounded prediction check": a real two-frame robot clip (Meta's own
franka_example_traj.npz) plus the real action between them. We predict the second
frame's embedding from the first + action and compare against the *actual* second
frame's embedding -- a real correctness number, not a demo trick. We also sweep a
small grid of candidate (dx,dy) actions and plot the resulting prediction error as a
heatmap, same methodology as Meta's own energy_landscape_example.ipynb, so you can see
the real action sit at (or near) the low-error point.

Tab 2, "Make it dance": since there's no camera footage for a made-up dance, this
chains the predictor's own output back in as the next "observed" frame -- literally the
same imagination-rollout Meta's CEM planner uses to evaluate candidate futures, just
driven by a chosen sequence of named move primitives (moves.py) instead of a real
trajectory. The per-step magnitude of change between consecutive imagined embeddings,
and the real measured forward-pass latency, drive the end-effector animation's color --
see robot_viz.py's docstring for the exact (tt-toplike-inspired) color law.

Tab 3, "CEM planning": a real optimizer, not a scripted move -- planning.py ports
Meta's own CEM (Cross-Entropy Method) action search to iteratively find the action that
best predicts the real frame 1 from frame 0, then reports how close it got to the
action that was actually taken.

Run locally against real Blackhole hardware (default): hold a gozer lease and set
TT_VISIBLE_DEVICES + TT_METAL_HOME first. Falls back to the CPU reference
implementation with --backend reference (what an HF Space without TT hardware runs).
"""

import argparse
import html as html_lib
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moves import MOVES, build_choreography  # noqa: E402
from planning import cem_search  # noqa: E402
from robot_viz import build_dance_figure, integrate_pose  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_TRAJ = REPO_ROOT / "reference" / "notebooks" / "franka_example_traj.npz"

# "Die Mensch-Maschine" palette: black ground, red/white accents, a geometric
# display face standing in for Kraftwerk's own Futura/Eurostile look (Orbitron is the
# closest well-supported Google Font to that family).
KRAFTWERK_FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&display=swap');"
KRAFTWERK_CSS = f"""
{KRAFTWERK_FONT_IMPORT}
.gradio-container {{ background: #000000 !important; }}
.gradio-container, .gradio-container * {{ font-family: 'Orbitron', monospace !important; }}
h1, h2, h3 {{ color: #ff2b2b !important; text-transform: uppercase; letter-spacing: 0.08em; }}
label span, .label-wrap span {{ color: #ff2b2b !important; text-transform: uppercase; letter-spacing: 0.04em; }}
button {{ text-transform: uppercase !important; letter-spacing: 0.06em !important; border-radius: 0 !important; }}
/* Gradio's own scoped button rule (e.g. `.primary.svelte-xxxx`) has 2 class
   selectors of specificity -- `button.primary` alone (1 class + 1 type) loses that
   tie even with !important. Match with >=2 classes so this reliably outranks it
   regardless of the build's scoped-class hash. */
button.lg.primary, button.lg.primary * {{ background: #ff2b2b !important; color: #000 !important; border: 2px solid #ff2b2b !important; font-weight: 700 !important; }}
.tabs button {{ color: #ff2b2b !important; }}
"""


def load_example():
    d = np.load(EXAMPLE_TRAJ)
    frames = d["observations"][0]  # [2, 256, 256, 3] uint8
    states = d["states"][0]  # [2, 7]
    return frames, states


def poses_to_action(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyz_diff = end[:3] - start[:3]
    r_s = Rotation.from_euler("xyz", start[3:6], degrees=False).as_matrix()
    r_e = Rotation.from_euler("xyz", end[3:6], degrees=False).as_matrix()
    theta_diff = Rotation.from_matrix(r_e @ r_s.T).as_euler("xyz", degrees=False)
    gripper_diff = end[6:7] - start[6:7]
    return np.concatenate([xyz_diff, theta_diff, gripper_diff]).astype(np.float32)


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).sqrt())


def run_grounded_check(backend, frames: np.ndarray, states: np.ndarray, grid_n: int = 5):
    """Real two-frame clip, real action. Returns (frame0, frame1, plotly heatmap fig,
    markdown report). See module docstring for the methodology."""
    import plotly.graph_objects as go

    f0, f1 = frames[0], frames[1]
    s0, s1 = states[0], states[1]
    action = poses_to_action(s0, s1)

    rep0 = backend.encode_frame(f0)
    rep1_actual = backend.encode_frame(f1)

    reps = rep0.unsqueeze(1)  # [1,1,HW,D]
    actions_t = torch.from_numpy(action).float().view(1, 1, 7)
    states_t = torch.from_numpy(s0).float().view(1, 1, 7)
    rep1_pred, latency_ms = backend.predict_step(reps, actions_t, states_t)
    real_error = l2(rep1_pred, rep1_actual)

    # (dx,dy) grid sweep around the real action, same methodology as Meta's own
    # energy_landscape_example.ipynb (nsamples=5, grid_size~0.075).
    grid = np.linspace(-0.075, 0.075, grid_n)
    errors = np.zeros((grid_n, grid_n))
    for i, gdx in enumerate(grid):
        for j, gdy in enumerate(grid):
            a = action.copy()
            a[0] += gdx
            a[1] += gdy
            a_t = torch.from_numpy(a).float().view(1, 1, 7)
            rep_g, _ = backend.predict_step(reps, a_t, states_t)
            errors[j, i] = l2(rep_g, rep1_actual)

    fig = go.Figure(data=go.Heatmap(
        z=errors, x=grid, y=grid, colorscale="Turbo", colorbar=dict(title="prediction error"),
    ))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=14, color="white", symbol="x"),
                              name="real action"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0014", plot_bgcolor="#0a0014",
        title="prediction error vs. Δ(dx,dy) around the real action (white x = real action taken)",
        xaxis_title="Δdx", yaxis_title="Δdy",
    )

    report = (
        f"**Backend:** {backend.name}\n\n"
        f"**Real-action prediction error (L2, embedding space):** {real_error:.4f}\n\n"
        f"**Predictor forward-pass latency:** {latency_ms:.2f} ms\n\n"
        f"This is a real correctness check: frame 1's embedding is predicted from "
        f"frame 0 + the actual recorded action, then compared to frame 1's own "
        f"(real) encoding. Lower error near the white x in the heatmap means the "
        f"model's prediction landscape actually favors the action that was really "
        f"taken."
    )
    return f0, f1, fig, report


def run_cem_plan(backend, frames: np.ndarray, states: np.ndarray, cem_steps: int, samples: int, topk: int,
                  maxnorm: float = 0.12):
    """Runs the real CEM optimizer (planning.cem_search, a faithful port of Meta's own
    algorithm) to find the action that best predicts frame 1 from frame 0 -- then
    compares what it found against the actual recorded action. Returns (plotly
    convergence-curve figure, markdown report).

    `maxnorm` defaults to 0.12, not Meta's own 0.05 -- the real recorded action's
    translation (dx=0.092, dz=0.084) is itself larger than 0.05, so at that default
    every CEM candidate saturates at the clip boundary and "converges" toward the edge
    of the search space rather than the actual target. Verified empirically (checked
    both settings) before choosing 0.12 as the default here."""
    cem_steps, samples, topk = int(cem_steps), int(samples), int(topk)
    f0, f1 = frames[0], frames[1]
    s0, s1 = states[0], states[1]
    real_action = poses_to_action(s0, s1)

    rep0 = backend.encode_frame(f0)
    rep1_actual = backend.encode_frame(f1)

    found_action, history = cem_search(backend, rep0, s0, rep1_actual, cem_steps=cem_steps, samples=samples,
                                        topk=topk, maxnorm=maxnorm)

    # How good is the action CEM found, vs. the one that was really taken?
    actions_t = torch.from_numpy(found_action).float().view(1, 1, 7)
    states_t = torch.from_numpy(s0).float().view(1, 1, 7)
    rep_found, _ = backend.predict_step(rep0.unsqueeze(1), actions_t, states_t)
    found_error = l2(rep_found, rep1_actual)
    action_l2 = float(np.linalg.norm(found_action[:3] - real_action[:3]))  # translation only -- rotation isn't searched

    import plotly.graph_objects as go

    steps = [h["step"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=[h["mean_error"] for h in history], mode="lines+markers",
                              name="mean error (this iteration's samples)", line=dict(color="#ff8888")))
    fig.add_trace(go.Scatter(x=steps, y=[h["best_error"] for h in history], mode="lines+markers",
                              name="best error (this iteration's samples)", line=dict(color="#ff2b2b", width=3)))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(family="'Orbitron', monospace", color="#ff2b2b"),
        title="CEM convergence: prediction error toward the real frame 1, per iteration",
        xaxis_title="CEM iteration", yaxis_title="prediction error (L2, embedding space)",
    )

    report = (
        f"**Backend:** {backend.name} — {cem_steps} CEM iterations x {samples} samples "
        f"(top-{topk} kept per iteration, search radius maxnorm={maxnorm})\n\n"
        f"**CEM-found action (translation, gripper):** "
        f"dx={found_action[0]:.4f}, dy={found_action[1]:.4f}, dz={found_action[2]:.4f}, "
        f"gripper={found_action[6]:.4f}\n\n"
        f"**Real recorded action (same fields):** "
        f"dx={real_action[0]:.4f}, dy={real_action[1]:.4f}, dz={real_action[2]:.4f}, "
        f"gripper={real_action[6]:.4f}\n\n"
        f"**Translation distance between them:** {action_l2:.4f} m &nbsp;|&nbsp; "
        f"**CEM action's own prediction error:** {found_error:.4f}\n\n"
        f"This mirrors Meta's own CEM simplification: only translation + gripper are "
        f"searched, rotation is held at zero. The real action here has a small but "
        f"nonzero rotation, so CEM converges toward the best rotation-locked "
        f"approximation of it, not an exact match -- the translation distance above is "
        f"the honest measure of how close it got. If **maxnorm** is set smaller than "
        f"the real action's own magnitude, every candidate saturates at that clip "
        f"boundary instead of genuinely converging -- watch the convergence curve go "
        f"flat immediately if so."
    )
    return fig, report


def imagination_rollout(backend, start_frame: np.ndarray, start_pose: np.ndarray, actions: np.ndarray):
    """The core loop shared by every imagined-rollout tab: encode the one real
    starting frame, then chain the predictor's own output back in as the next
    'observed' frame for each action in turn -- exactly Meta's own CEM planner's
    imagination step (see world_model_wrapper.py::WorldModel.infer_next_action),
    just driven by a pre-chosen action sequence instead of a live optimizer.
    Returns (poses [N+1,7], energies [N], latencies_ms [N])."""
    n_steps = len(actions)
    poses = integrate_pose(start_pose, actions)
    rep0 = backend.encode_frame(start_frame)
    reps = rep0.unsqueeze(1)  # [1,1,HW,D]
    states_seq = [start_pose]
    energies, latencies = [], []
    for i in range(n_steps):
        actions_t = torch.from_numpy(actions[: i + 1]).float().unsqueeze(0)  # [1,i+1,7]
        states_t = torch.from_numpy(np.stack(states_seq)).float().unsqueeze(0)  # [1,i+1,7]
        next_rep, latency_ms = backend.predict_step(reps, actions_t, states_t)
        energies.append(l2(next_rep, reps[:, -1]))
        latencies.append(latency_ms)
        reps = torch.cat([reps, next_rep.unsqueeze(1)], dim=1)
        states_seq.append(poses[i + 1])
    return poses, np.array(energies), np.array(latencies)


def run_choreography(backend, frames: np.ndarray, states: np.ndarray, sequence: str, steps_per_move: int):
    """Chains named moves (see moves.py) into one action sequence, then runs the same
    imagination rollout every 'imagined' tab uses. Returns (plotly animated figure,
    markdown report)."""
    steps_per_move = int(steps_per_move)
    actions, segments = build_choreography(sequence, steps_per_move)
    if len(actions) == 0:
        return None, (
            f"No recognized moves in `{sequence}`. Known moves: {', '.join(MOVES)}."
        )
    start_pose = states[0].copy()
    poses, energies, latencies = imagination_rollout(backend, frames[0], start_pose, actions)

    fig = build_dance_figure(poses, energies, latencies)
    segment_lines = "\n".join(f"- **{name}**: steps {start}–{end}" for name, start, end in segments)
    report = (
        f"**Backend:** {backend.name} — {len(actions)} imagined steps across "
        f"{len(segments)} moves, {np.mean(latencies):.2f} ms/step avg predictor latency\n\n"
        f"{segment_lines}\n\n"
        f"Color = per-step imagined-embedding change (hue, biased red = bigger "
        f"change) and real measured latency (brightness). No ground-truth video "
        f"exists for this choreography -- the model is chaining its own predicted "
        f"embeddings forward across every move, exactly how CEM planning imagines "
        f"candidate futures in Meta's own reference implementation."
    )
    return fig, report


def build_app(backend):
    frames, states = load_example()

    with gr.Blocks(title="V-JEPA2-AC on Blackhole") as demo:
        gr.Markdown(
            "# 🤖 WIR SIND DIE ROBOTER\n"
            "### V-JEPA2-AC — MENSCH-MASCHINE WELTMODELL, AUF TENSTORRENT BLACKHOLE\n"
            f"BACKEND: **{backend.name}**. DIESES MODELL SAGT EINBETTUNGEN VORAUS, "
            "KEINE PIXEL — alles Visuelle unten ist entweder eine echte "
            "Korrektheitsprüfung oder ein klar gekennzeichneter imaginierter Ablauf, "
            "niemals erzeugtes Video."
        )
        with gr.Tab("GRUNDLAGEN-PRÜFUNG // GROUNDED CHECK"):
            btn1 = gr.Button("SYSTEM AKTIVIEREN // RUN ON REAL CLIP", variant="primary")
            with gr.Row():
                img0 = gr.Image(label="BILD 0 // FRAME 0 (real)")
                img1 = gr.Image(label="BILD 1 // FRAME 1 (real)")
            plot1 = gr.Plot(label="FEHLERLANDSCHAFT // PREDICTION-ERROR LANDSCAPE")
            report1 = gr.Markdown()
            btn1.click(lambda: run_grounded_check(backend, frames, states), outputs=[img0, img1, plot1, report1])
        with gr.Tab("TANZEN // DANCE"):
            gr.Markdown(
                "_Jeder Schritt wächst den Kontext um ein Bild -- Schritt N hat eine "
                "andere Sequenzlänge als Schritt N-1, also kompiliert das "
                "Blackhole-Backend beim **ersten** Lauf neue Kernels für JEDEN "
                "Schrittindex 1..N separat (Sekunden pro neuer Länge, nicht einmalig "
                "insgesamt) -- eine 30-Schritt-Choreografie kann beim ersten Mal "
                "mehrere Minuten dauern. Danach sind alle diese Längen im "
                "TTNN-Kernel-Cache und wiederholte Läufe sind schnell. Kurz anfangen, "
                "dann verlängern._\n\n"
                f"**BEKANNTE SCHRITTE // KNOWN MOVES:** {', '.join(MOVES)}"
            )
            sequence = gr.Textbox(
                value="ACHT SPIN VERBEUGUNG",
                label="CHOREOGRAFIE // CHOREOGRAPHY (move names, space-separated, repeats allowed)",
            )
            steps_per_move = gr.Slider(3, 10, value=4, step=1, label="SCHRITTE PRO BEWEGUNG // STEPS PER MOVE")
            btn2 = gr.Button("TANZ BERECHNEN // COMPUTE DANCE", variant="primary")
            # gr.Plot never calls Plotly.addFrames (Play button/slider render but do
            # nothing -- see the earlier fix). gr.HTML looked like the answer since
            # fig.to_html() DOES include that call, but gr.HTML renders via Svelte's
            # {@html ...}, which -- like any innerHTML-style insertion -- never
            # executes embedded <script> tags at all (a browser/Svelte limitation, not
            # a Gradio bug): the plot didn't just fail to animate, it failed to render
            # at all. An <iframe srcdoc="..."> is a genuinely separate document parse,
            # where scripts DO run normally, and is the standard way to embed a
            # self-contained interactive widget inside a page that won't execute
            # injected scripts for you.
            plot2 = gr.HTML(label="imagined end-effector trajectory")
            report2 = gr.Markdown()

            def run_choreography_html(*a):
                fig, report = run_choreography(backend, frames, states, *a)
                if fig is None:
                    return "", report
                doc = fig.to_html(full_html=True, include_plotlyjs="cdn")
                doc = doc.replace("<head>", f"<head><style>{KRAFTWERK_FONT_IMPORT} body{{margin:0}}</style>", 1)
                iframe = (
                    f'<iframe srcdoc="{html_lib.escape(doc, quote=True)}" '
                    'style="width:100%;height:580px;border:none;background:#000000;"></iframe>'
                )
                return iframe, report

            btn2.click(run_choreography_html, inputs=[sequence, steps_per_move], outputs=[plot2, report2])
        with gr.Tab("PLANEN // CEM PLANNING"):
            gr.Markdown(
                "_Statt einer festen Bewegung sucht hier ein echter CEM-Optimierer "
                "(portiert von Metas eigenem `mpc_utils.cem`) iterativ nach der Aktion, "
                "die Bild 1 aus Bild 0 am besten vorhersagt -- und vergleicht das "
                "Ergebnis mit der tatsächlich aufgezeichneten Aktion._"
            )
            with gr.Row():
                cem_steps = gr.Slider(2, 15, value=6, step=1, label="CEM-ITERATIONEN // CEM ITERATIONS")
                cem_samples = gr.Slider(4, 32, value=12, step=1, label="STICHPROBEN // SAMPLES / ITERATION")
                cem_topk = gr.Slider(2, 12, value=4, step=1, label="TOP-K")
                cem_maxnorm = gr.Slider(0.02, 0.15, value=0.12, step=0.01, label="SUCHRADIUS // SEARCH RADIUS (maxnorm, m)")
            btn3 = gr.Button("CEM STARTEN // RUN CEM", variant="primary")
            plot3 = gr.Plot(label="CEM-KONVERGENZ // CEM CONVERGENCE")
            report3 = gr.Markdown()
            btn3.click(lambda *a: run_cem_plan(backend, frames, states, *a),
                       inputs=[cem_steps, cem_samples, cem_topk, cem_maxnorm], outputs=[plot3, report3])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ttnn", "reference"], default="ttnn")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.backend == "ttnn":
        from backends import TTNNBackend
        backend = TTNNBackend(device_id=args.device_id)
    else:
        from backends import ReferenceBackend
        backend = ReferenceBackend()

    demo = build_app(backend)
    try:
        demo.launch(share=args.share, theme=gr.themes.Base(primary_hue="red"), css=KRAFTWERK_CSS)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
