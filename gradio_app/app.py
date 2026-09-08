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
from pathlib import Path

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moves import MOVES, build_choreography, build_show  # noqa: E402
from planning import cem_search, plan_step  # noqa: E402
from robot_viz import build_dance_figure, build_multi_dancer_figure, integrate_pose  # noqa: E402

# Injected into the iframe's own document (not the outer Gradio page) after the
# figure's HTML: polls until Plotly.addFrames has actually populated frames (rather
# than assuming a fixed ordering against the newPlot(...).then(addFrames) promise
# chain already in fig.to_html()'s output), then plays once and replays from the start
# every time the animation finishes -- turning "click generate, watch once, stop" into
# a show that runs until the tab is closed.
AUTO_LOOP_SCRIPT = """
<script>
(function() {
  function tryPlay() {
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd || !gd._transitionData || !gd._transitionData._frames || !gd._transitionData._frames.length) {
      setTimeout(tryPlay, 150);
      return;
    }
    function playForward() {
      Plotly.animate(gd, null, {frame: {duration: 220, redraw: true}, transition: {duration: 0}, mode: 'immediate'});
    }
    function restart() {
      // Two things Plotly needs here, found the hard way: (1) Plotly.animate(gd, null,
      // ...) after already reaching the last frame does not reliably restart from the
      // top on its own -- explicitly seek back to frame '0' first, the same call the
      // slider itself makes. (2) calling Plotly.animate() again SYNCHRONOUSLY from
      // inside its own 'plotly_animated' completion handler throws an internal
      // re-entrancy rejection (Plotly hasn't finished settling the previous call's
      // state yet) -- deferring one tick with setTimeout lets that settle first.
      setTimeout(function() {
        Plotly.animate(gd, ['0'], {frame: {duration: 0, redraw: true}, transition: {duration: 0}, mode: 'immediate'})
          .then(function() { setTimeout(playForward, 50); });
      }, 50);
    }
    gd.on('plotly_animated', restart);
    playForward();
  }
  tryPlay();
})();
</script>
"""

# Show-tab-only: real Web Audio sonification, one "voice" per act palette (see
# robot_viz.PALETTE_VOICES). Every voice's pitch is driven by that frame's real
# energy_norm and its brightness/rate by real latency_norm -- the same "every signal
# here is real" honesty the color law already has, now extended to sound. A steady
# low motorikPulse plays under every non-silent frame (Kraftwerk's motorik rhythm);
# the palette's own voice is the melodic/textural layer riding on top of it.
#
# This replaces the single continuous `Plotly.animate(gd, null, ...)` call the plain
# AUTO_LOOP_SCRIPT uses with a step-by-step loop (one frame at a time), so each frame
# transition can trigger its own matching audio event -- audio and visuals advance
# together, one animate-promise at a time, looping via index wraparound instead of
# the plain script's explicit "seek back to frame 0" restart dance.
SHOW_AUTO_LOOP_SCRIPT = """
<script>
(function() {
  var audioCtx = null;
  function ensureAudio() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return audioCtx;
  }

  var STEP_DUR = 0.22; // matches the 220ms/frame animation cadence below

  function motorikPulse(c, t) {
    var osc = c.createOscillator(), gain = c.createGain();
    osc.type = 'sine'; osc.frequency.value = 55;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.3, t + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.09);
    osc.connect(gain).connect(c.destination);
    osc.start(t); osc.stop(t + 0.1);
  }

  function leadTone(c, t, dur, freq, filtFreq, gainPeak) {
    var osc = c.createOscillator(), filt = c.createBiquadFilter(), gain = c.createGain();
    osc.type = 'sawtooth'; osc.frequency.value = freq;
    filt.type = 'lowpass'; filt.frequency.value = filtFreq;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(gainPeak, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.connect(filt).connect(gain).connect(c.destination);
    osc.start(t); osc.stop(t + dur);
  }

  function noiseBurst(c, t, dur, gainPeak, filtFreq) {
    var bufSize = Math.max(1, Math.floor(c.sampleRate * dur));
    var buf = c.createBuffer(1, bufSize, c.sampleRate);
    var data = buf.getChannelData(0);
    for (var i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1;
    var src = c.createBufferSource(); src.buffer = buf;
    var filt = c.createBiquadFilter(); filt.type = 'bandpass'; filt.frequency.value = filtFreq; filt.Q.value = 2.5;
    var gain = c.createGain();
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(gainPeak, t + 0.003);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(filt).connect(gain).connect(c.destination);
    src.start(t); src.stop(t + dur);
  }

  var VOICES = {
    motorik: function(c, t, m) {
      leadTone(c, t, STEP_DUR * 0.9, 220 + m.energy_norm * 440, 400 + m.latency_norm * 3000, 0.16);
    },
    ratchet: function(c, t, m) {
      var n = 2 + Math.round(m.energy_norm * 5);
      for (var k = 0; k < n; k++) noiseBurst(c, t + k * (STEP_DUR / n) * 0.8, 0.02, 0.2, 1200 + m.energy_norm * 2500);
    },
    stab: function(c, t) { leadTone(c, t, 0.12, 660, 4000, 0.28); },
    soft: function(c, t, m) {
      leadTone(c, t, STEP_DUR * 0.8, 160 + m.energy_norm * 120, 500 + m.latency_norm * 800, 0.08);
    },
    ticks: function(c, t, m) {
      var osc = c.createOscillator(), gain = c.createGain();
      osc.type = 'sine'; osc.frequency.value = 1800 + m.energy_norm * 1200;
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.04, t + 0.002);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.03);
      osc.connect(gain).connect(c.destination);
      osc.start(t); osc.stop(t + 0.04);
    },
    arpeggio: function(c, t, m) {
      var notes = [0, 4, 7, 12];
      var base = 300 + m.energy_norm * 300;
      notes.forEach(function(semi, idx) {
        var f = base * Math.pow(2, semi / 12);
        leadTone(c, t + idx * (STEP_DUR / notes.length) * 0.8, STEP_DUR / notes.length,
                 f, 2000 + m.latency_norm * 3000, 0.12);
      });
    },
    chime: function(c, t, m) {
      var osc = c.createOscillator(), gain = c.createGain();
      osc.type = 'triangle'; osc.frequency.value = 500 + m.energy_norm * 500;
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.15, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, t + STEP_DUR * 1.5);
      osc.connect(gain).connect(c.destination);
      osc.start(t); osc.stop(t + STEP_DUR * 1.5);
    },
    silence: function() {},
  };

  function triggerAudio(meta) {
    if (!audioCtx || !meta || meta.voice === 'silence') return;
    var t = audioCtx.currentTime + 0.01;
    motorikPulse(audioCtx, t);
    (VOICES[meta.voice] || VOICES.motorik)(audioCtx, t, meta);
  }

  function tryPlay() {
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd || !gd._transitionData || !gd._transitionData._frames || !gd._transitionData._frames.length) {
      setTimeout(tryPlay, 150);
      return;
    }
    var idx = 0;
    function playFrame() {
      var meta = FRAME_AUDIO[idx];
      triggerAudio(meta);
      var duration = (meta && meta.duration_ms) || 220;
      Plotly.animate(gd, [String(idx)], {frame: {duration: duration, redraw: true}, transition: {duration: 0}, mode: 'immediate'})
        .then(function() {
          idx = (idx + 1) % FRAME_AUDIO.length;
          setTimeout(playFrame, 0);
        });
    }
    playFrame();
  }
  tryPlay();

  // A fresh iframe document needs its own user gesture to unlock a new AudioContext
  // -- the outer Gradio "Start the Show" click doesn't count for this document.
  var btn = document.createElement('button');
  btn.textContent = '\N{SPEAKER WITH THREE SOUND WAVES}️ Enable Sound';
  btn.style.cssText = 'position:fixed;top:10px;right:10px;z-index:1000;padding:8px 14px;'
    + 'background:#1a0000;color:#ff8888;border:1px solid #ff2b2b;border-radius:6px;'
    + "font-family:'Orbitron',monospace;font-size:12px;cursor:pointer;";
  btn.onclick = function() { ensureAudio(); btn.remove(); };
  document.body.appendChild(btn);
})();
</script>
"""

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_TRAJ = REPO_ROOT / "reference" / "notebooks" / "franka_example_traj.npz"

# "Die Mensch-Maschine" palette: black ground, red/white accents, a geometric
# display face standing in for Kraftwerk's own Futura/Eurostile look (Orbitron is the
# closest well-supported Google Font to that family). This stays exactly as-is for the
# show/dance iframes themselves (see wrap_iframe) -- it's the performance's own
# identity, not the surrounding app chrome.
KRAFTWERK_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&display=swap');"
)

# The surrounding Gradio UI (outside the iframes) uses tt-toplike's "Grayskull" theme
# concept (see ~/code/tt-toplike/src/ui/colors.rs::grayskull_rgb): the rainbow
# collapsed to "a thousand shades of grey", with hot pink as the one deliberately
# saturated accent color -- a moody, monochrome hacker-tool look, distinct from both
# the show's own Kraftwerk red/black and a first attempt at a lighter, literal
# brand-color theme that didn't read right. Neue Haas Unica Pro / Degular (the actual
# TT brand type) aren't freely licensed for web embedding, so IBM Plex Sans stands in:
# a clean, geometric, neutral face, picked specifically to avoid the generic
# Inter/Space-Grotesk "AI-default" look.
TT_BRAND_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');"
)
TT_BRAND_CSS = f"""
{TT_BRAND_FONT_IMPORT}
.gradio-container {{ background: #0d0d0d !important; }}
.gradio-container, .gradio-container * {{ font-family: 'IBM Plex Sans', sans-serif !important; }}
/* Same specificity lesson as the button fix below: Gradio's own scoped text-color
   rules can outrank a plain single-class selector even with !important, so match
   broadly and use !important throughout. */
.gradio-container p, .gradio-container span, .gradio-container .prose, .gradio-container .prose * {{
  color: #cfcfcf !important;
}}
h1, h2, h3 {{ color: #eaeaea !important; letter-spacing: -0.01em; }}
h1 {{ color: #ff4fa3 !important; }}
label span, .label-wrap span {{ color: #8a8a8a !important; font-weight: 500; }}
button {{ border-radius: 10px !important; font-weight: 500 !important; background: #1a1a1a !important; color: #cfcfcf !important; border: 1px solid #333333 !important; }}
/* Gradio's own scoped button rule (e.g. `.primary.svelte-xxxx`) has 2 class
   selectors of specificity -- `button.primary` alone (1 class + 1 type) loses that
   tie even with !important. Match with >=2 classes so this reliably outranks it
   regardless of the build's scoped-class hash. */
button.lg.primary, button.lg.primary * {{ background: #ff4fa3 !important; color: #0d0d0d !important; border: none !important; font-weight: 600 !important; }}
.tabs button {{ color: #8a8a8a !important; font-weight: 500; }}
.tabs button.selected {{ color: #ff4fa3 !important; }}
.block {{ background: #161616 !important; border-color: #2a2a2a !important; border-radius: 12px !important; }}
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

    fig = go.Figure(
        data=go.Heatmap(
            z=errors,
            x=grid,
            y=grid,
            colorscale="Turbo",
            colorbar=dict(title="prediction error"),
        )
    )
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=14, color="white", symbol="x"), name="real action")
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0014",
        plot_bgcolor="#0a0014",
        title="prediction error vs. Δ(dx,dy) around the real action (white x = real action taken)",
        xaxis_title="Δdx",
        yaxis_title="Δdy",
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


def run_cem_plan(
    backend, frames: np.ndarray, states: np.ndarray, cem_steps: int, samples: int, topk: int, maxnorm: float = 0.12
):
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

    found_action, history = cem_search(
        backend, rep0, s0, rep1_actual, cem_steps=cem_steps, samples=samples, topk=topk, maxnorm=maxnorm
    )

    # How good is the action CEM found, vs. the one that was really taken?
    actions_t = torch.from_numpy(found_action).float().view(1, 1, 7)
    states_t = torch.from_numpy(s0).float().view(1, 1, 7)
    rep_found, _ = backend.predict_step(rep0.unsqueeze(1), actions_t, states_t)
    found_error = l2(rep_found, rep1_actual)
    action_l2 = float(np.linalg.norm(found_action[:3] - real_action[:3]))  # translation only -- rotation isn't searched

    import plotly.graph_objects as go

    steps = [h["step"] for h in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[h["mean_error"] for h in history],
            mode="lines+markers",
            name="mean error (this iteration's samples)",
            line=dict(color="#ff8888"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[h["best_error"] for h in history],
            mode="lines+markers",
            name="best error (this iteration's samples)",
            line=dict(color="#ff2b2b", width=3),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(family="'Orbitron', monospace", color="#ff2b2b"),
        title="CEM convergence: prediction error toward the real frame 1, per iteration",
        xaxis_title="CEM iteration",
        yaxis_title="prediction error (L2, embedding space)",
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


def imagination_rollout(
    backend, start_frame: np.ndarray, start_pose: np.ndarray, actions: np.ndarray, rep0: torch.Tensor | None = None
):
    """The core loop shared by every imagined-rollout tab: encode the one real
    starting frame (unless a precomputed `rep0` is passed in, e.g. when several
    independent rollouts share the same starting frame -- see run_show), then chain
    the predictor's own output back in as the next 'observed' frame for each action in
    turn -- exactly Meta's own CEM planner's imagination step (see
    world_model_wrapper.py::WorldModel.infer_next_action), just driven by a
    pre-chosen action sequence instead of a live optimizer.
    Returns (poses [N+1,7], energies [N], latencies_ms [N])."""
    n_steps = len(actions)
    poses = integrate_pose(start_pose, actions)
    if rep0 is None:
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


def cem_imagination_rollout(
    backend, start_frame: np.ndarray, start_pose: np.ndarray, target_actions: np.ndarray,
    rep0: torch.Tensor | None = None, cem_steps: int = 6, samples: int = 12, topk: int = 4,
    maxnorm: float = 0.12,
):
    """Sibling to imagination_rollout: instead of executing each of `target_actions`
    verbatim, each is used only as goal-probing INTENT for planning.plan_step, which
    lets CEM search for a real action reaching near that goal and executes what it
    actually found -- the rendered trajectory is the model's own choice, target_actions
    only ever supply intent. Same chaining, same return shape (poses [N+1,7],
    energies [N], latencies_ms [N]), so it's a straight swap in run_choreography --
    just slower, since every step now costs cem_steps+2 backend calls instead of 1."""
    if rep0 is None:
        rep0 = backend.encode_frame(start_frame)
    reps = rep0.unsqueeze(1)  # [1,1,HW,D]
    cur_pose = start_pose.copy()
    poses = [start_pose.copy()]
    prior_actions: list = []
    states_seq = [start_pose]
    energies, latencies = [], []
    for target in target_actions:
        found_action, next_rep, energy, latency_ms = plan_step(
            backend, reps, prior_actions, states_seq, cur_pose, target,
            cem_steps=cem_steps, samples=samples, topk=topk, maxnorm=maxnorm,
        )
        energies.append(energy)
        latencies.append(latency_ms)
        reps = torch.cat([reps, next_rep.unsqueeze(1)], dim=1)
        prior_actions.append(found_action)
        cur_pose = integrate_pose(cur_pose, found_action.reshape(1, 7))[-1]
        states_seq.append(cur_pose)
        poses.append(cur_pose.copy())
    return np.stack(poses), np.array(energies), np.array(latencies)


def run_choreography(backend, frames: np.ndarray, states: np.ndarray, sequence: str, steps_per_move: int,
                      use_planning: bool = False):
    """Chains named moves (see moves.py) into one action sequence. By default plays
    each action back verbatim (imagination_rollout) -- the same imagined rollout every
    'imagined' tab uses. With `use_planning=True`, each move's action is instead used
    only as goal-probing intent for cem_imagination_rollout: CEM searches for a real
    action reaching near that intent, and the model's own choice is what actually gets
    executed and rendered -- slower, but genuinely model-chosen rather than scripted.
    Returns (plotly animated figure, markdown report)."""
    steps_per_move = int(steps_per_move)
    actions, segments = build_choreography(sequence, steps_per_move)
    if len(actions) == 0:
        return None, (f"No recognized moves in `{sequence}`. Known moves: {', '.join(MOVES)}.")
    start_pose = states[0].copy()
    if use_planning:
        poses, energies, latencies = cem_imagination_rollout(backend, frames[0], start_pose, actions)
    else:
        poses, energies, latencies = imagination_rollout(backend, frames[0], start_pose, actions)

    fig = build_dance_figure(poses, energies, latencies)
    segment_table = "| # | Move | Steps |\n|---|---|---|\n" + "\n".join(
        f"| {i + 1} | {name} | {start}–{end} |" for i, (name, start, end) in enumerate(segments)
    )
    mode_note = (
        "**Plan with CEM** is on: each move below supplied only the *intent* -- a real "
        "CEM search (`planning.plan_step`) found the action actually executed at every "
        "step, and that found action, not the authored curve, is what's rendered above."
        if use_planning else
        "Color = per-step imagined-embedding change (hue, biased red = bigger "
        "change) and real measured latency (brightness). No ground-truth video "
        "exists for this choreography -- the model is chaining its own predicted "
        "embeddings forward across every move, exactly how CEM planning imagines "
        "candidate futures in Meta's own reference implementation."
    )
    report = (
        f"**Backend:** {backend.name} — {len(actions)} imagined steps across "
        f"{len(segments)} moves, {np.mean(latencies):.2f} ms/step avg predictor latency\n\n"
        f"{segment_table}\n\n"
        f"{mode_note}"
    )
    return fig, report


STAGE_SPACING = 0.55  # meters between each act's own spot on the shared stage


def run_show(backend, frames: np.ndarray, states: np.ndarray):
    """The full seven-act show (moves.build_show): INSTANT KRAFTWERK, Careful with
    that Ax Eugene, Tangerine Ratchet, Poppin and Lockin, Your Name on a Grain of Rice,
    Laser Cats Cutting a Rug, and the XOXO TT finale -- staged as separate dancers, not
    one continuous trajectory. Consecutive act_segments sharing a display name (e.g.
    the Ax Eugene build + its snap) are grouped into one 'slot': each slot gets its own
    independent imagination rollout, always starting fresh from the same one real
    encoded frame (never chained from a previous act's ending pose), then placed at
    its own fixed spot on the stage. Earlier slots stay frozen in their final pose as
    the show moves on to the next, instead of one dancer relocating through all seven
    acts -- and since no single rollout call now needs more than the longest ONE act's
    own step count as context (not the cumulative total across the whole show), this
    is also considerably safer on device memory than the one-continuous-rollout
    version was (see functional_predictor.py's get_rope_and_mask docstring for why an
    ever-growing single context is the actual DRAM-exhaustion risk)."""
    actions, pen_up, labels, act_segments = build_show()
    start_pose = states[0].copy()
    rep0 = backend.encode_frame(frames[0])  # shared across every slot's independent rollout

    slot_groups = []
    for seg in act_segments:
        if slot_groups and slot_groups[-1]["name"] == seg["name"]:
            slot_groups[-1]["segments"].append(seg)
        else:
            slot_groups.append({"name": seg["name"], "segments": [seg]})

    slots = []
    for i, group in enumerate(slot_groups):
        lo, hi = group["segments"][0]["start"], group["segments"][-1]["end"]
        poses, energies, latencies = imagination_rollout(backend, frames[0], start_pose, actions[lo:hi], rep0=rep0)
        poses = poses.copy()
        poses[:, :3] += np.array([0.0, i * STAGE_SPACING, 0.0], dtype=np.float32)
        palette_segments = [(seg["palette"], seg["start"] - lo, seg["end"] - lo) for seg in group["segments"]]
        slots.append(
            dict(
                name=group["name"],
                poses=poses,
                energies=energies,
                latencies_ms=latencies,
                pen_up=pen_up[lo:hi],
                labels=labels[lo:hi],
                palette_segments=palette_segments,
                zoom=group["segments"][0]["zoom"],
                camera_bias=group["segments"][0]["camera_bias"],
                speed=group["segments"][0]["speed"],
            )
        )

    fig, frame_audio = build_multi_dancer_figure(slots)

    total_steps = sum(len(s["energies"]) for s in slots)
    longest = max(len(s["energies"]) for s in slots)
    all_latencies = np.concatenate([s["latencies_ms"] for s in slots])
    act_table = "| # | Act | Steps | Palette | Avg latency |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {i + 1} | {s['name']} | {len(s['energies'])} | {s['palette_segments'][0][0]} | "
        f"{np.mean(s['latencies_ms']):.1f} ms |"
        for i, s in enumerate(slots)
    )
    report = (
        f"**Backend:** {backend.name} — {total_steps} imagined steps across "
        f"{len(slots)} independently-staged dancers (longest single rollout: "
        f"{longest} steps), {np.mean(all_latencies):.2f} ms/step avg predictor latency\n\n"
        f"{act_table}\n\n"
        f"Plays once through, then loops from the top automatically -- no need to "
        f"press anything again. The **first** run compiles new TTNN kernels for every "
        f"distinct step count any single dancer reaches (a real one-time cost, seconds "
        f"per new length); every loop after that reuses the compiled kernels and is "
        f"fast."
    )
    return fig, frame_audio, report


def wrap_iframe(fig, height: int = 600, auto_loop: bool = False, frame_audio: list | None = None) -> str:
    """Wraps a plotly Figure as a self-contained <iframe srcdoc="...">, with the
    Orbitron import carried along (the iframe's document doesn't inherit the outer
    page's <head>). See the TANZEN tab's original comment for why an iframe is needed
    at all instead of gr.Plot/gr.HTML directly. `auto_loop` appends AUTO_LOOP_SCRIPT so
    the animation restarts itself forever instead of stopping after one pass.

    `frame_audio` (Show tab only): the per-frame sonification metadata from
    build_multi_dancer_figure. When given, it takes over looping entirely -- injects
    it as `FRAME_AUDIO` plus SHOW_AUTO_LOOP_SCRIPT (the audio-aware, step-by-step
    player) instead of the plain AUTO_LOOP_SCRIPT, and `auto_loop` is ignored."""
    doc = fig.to_html(full_html=True, include_plotlyjs="cdn")
    doc = doc.replace("<head>", f"<head><style>{KRAFTWERK_FONT_IMPORT} body{{margin:0}}</style>", 1)
    if frame_audio is not None:
        import json

        script = f"<script>const FRAME_AUDIO = {json.dumps(frame_audio)};</script>" + SHOW_AUTO_LOOP_SCRIPT
        doc = doc.replace("</body>", script + "</body>", 1)
    elif auto_loop:
        doc = doc.replace("</body>", AUTO_LOOP_SCRIPT + "</body>", 1)
    return (
        f'<iframe srcdoc="{html_lib.escape(doc, quote=True)}" '
        f'style="width:100%;height:{height}px;border:none;background:#000000;"></iframe>'
    )



# TTNNBackend wraps a single ttnn.Device handle, which is not safe for concurrent
# calls from multiple threads -- Gradio queues each button's own repeat clicks
# (concurrency_limit=1 is per-listener by default), but nothing stops two DIFFERENT
# tabs' handlers from running at once otherwise. Sharing one concurrency_id across
# every backend-touching listener below serializes them against each other too, app-
# wide, regardless of which tab. Discovered the hard way: three tabs' handlers ended
# up mid-flight on the same device simultaneously (py-spy showed Show, Grounded Check,
# and CEM Planning all blocked inside device calls at once), which from the outside
# looked exactly like a hang.
BACKEND_CONCURRENCY_ID = "ttnn-backend"


def build_app(backend):
    frames, states = load_example()

    with gr.Blocks(
        title="V-JEPA2-AC on Blackhole",
        theme=gr.themes.Base(primary_hue="pink", neutral_hue="gray"),
        css=TT_BRAND_CSS,
    ) as demo:
        gr.Markdown(
            "# We Are The Robots\n"
            "### V-JEPA2-AC — an action-conditioned world model, on Tenstorrent Blackhole\n"
            f"Backend: **{backend.name}**. This model predicts *embeddings*, not pixels "
            "— everything visual below is either a real correctness check or a clearly "
            "labeled imagined rollout, never generated video."
        )
        with gr.Tab("★ The Show"):
            gr.Markdown(
                "_Seven acts, one continuous show: Instant Kraftwerk → Careful with "
                "that Ax, Eugene → Tangerine Ratchet → Poppin and Lockin → Your Name "
                "on a Grain of Rice → Laser Cats Cutting a Rug → XOXO TT — then loops "
                "from the top, automatically, forever. Each act gets its own color, "
                "camera, and stage spot; earlier acts stay frozen in place as later "
                "ones take the spotlight. The **first** run compiles new kernels for "
                "every distinct step count (a real one-time cost, seconds per length); "
                "every loop after that is fast._"
            )
            btn0 = gr.Button("▶ Start the Show", variant="primary")
            plot0 = gr.HTML(label="the show")
            report0 = gr.Markdown()

            def run_show_html():
                fig, frame_audio, report = run_show(backend, frames, states)
                return wrap_iframe(fig, height=620, frame_audio=frame_audio), report

            btn0.click(run_show_html, outputs=[plot0, report0],
                       concurrency_id=BACKEND_CONCURRENCY_ID, concurrency_limit=1)
        with gr.Tab("Grounded Check"):
            btn1 = gr.Button("Run on the Real Clip", variant="primary")
            with gr.Row():
                img0 = gr.Image(label="Frame 0 (real)")
                img1 = gr.Image(label="Frame 1 (real)")
            plot1 = gr.Plot(label="Prediction-Error Landscape")
            report1 = gr.Markdown()
            btn1.click(lambda: run_grounded_check(backend, frames, states), outputs=[img0, img1, plot1, report1],
                       concurrency_id=BACKEND_CONCURRENCY_ID, concurrency_limit=1)
        with gr.Tab("Dance"):
            gr.Markdown(
                "_Every step grows the context by one frame -- step N is a different "
                "sequence length than step N-1, so the Blackhole backend compiles new "
                "kernels for EACH step index 1..N separately on the **first** run "
                "(seconds per new length, not a one-time total) -- a 30-step "
                "choreography can take a few minutes the first time. After that, "
                "every one of those lengths is in the TTNN kernel cache and repeat "
                "runs are fast. Start short, then lengthen it._\n\n"
                "| Move | What it does |\n|---|---|\n"
                + "\n".join(
                    f"| **{name}** | {' '.join(fn.__doc__.split(' -- ', 1)[-1].split()).split('. ')[0].rstrip('.')} |"
                    for name, fn in MOVES.items()
                )
            )
            sequence = gr.Textbox(
                value="FIGURE8 SPIN BOW",
                label="Choreography (move names, space-separated, repeats allowed)",
            )
            steps_per_move = gr.Slider(3, 10, value=4, step=1, label="Steps per Move")
            plan_toggle = gr.Checkbox(
                value=False,
                label="Plan with CEM (slower, model-chosen -- each move supplies intent only; "
                      "a real CEM search finds the action actually executed at every step)",
            )
            btn2 = gr.Button("Compute Dance", variant="primary")
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
                return wrap_iframe(fig, height=580, auto_loop=True), report

            btn2.click(run_choreography_html, inputs=[sequence, steps_per_move, plan_toggle],
                       outputs=[plot2, report2],
                       concurrency_id=BACKEND_CONCURRENCY_ID, concurrency_limit=1)
        with gr.Tab("CEM Planning"):
            gr.Markdown(
                "_Instead of a scripted move, a real CEM optimizer (ported from Meta's "
                "own `mpc_utils.cem`) iteratively searches for the action that best "
                "predicts frame 1 from frame 0 -- then compares what it found against "
                "the action that was actually recorded._"
            )
            with gr.Row():
                cem_steps = gr.Slider(2, 15, value=6, step=1, label="CEM Iterations")
                cem_samples = gr.Slider(4, 32, value=12, step=1, label="Samples / Iteration")
                cem_topk = gr.Slider(2, 12, value=4, step=1, label="Top-K")
                cem_maxnorm = gr.Slider(0.02, 0.15, value=0.12, step=0.01, label="Search Radius (maxnorm, m)")
            btn3 = gr.Button("Run CEM", variant="primary")
            plot3 = gr.Plot(label="CEM Convergence")
            report3 = gr.Markdown()
            btn3.click(
                lambda *a: run_cem_plan(backend, frames, states, *a),
                inputs=[cem_steps, cem_samples, cem_topk, cem_maxnorm],
                outputs=[plot3, report3],
                concurrency_id=BACKEND_CONCURRENCY_ID, concurrency_limit=1,
            )
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
        demo.launch(share=args.share)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
