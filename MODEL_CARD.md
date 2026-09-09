# Model Card: V-JEPA2-AC on Tenstorrent Blackhole (this repo's TTNN port)

Follows the structure of [Mitchell et al., "Model Cards for Model Reporting"
(2019)](https://www.semanticscholar.org/paper/Model-Cards-for-Model-Reporting-Mitchell-Wu/7365f887c938ca21a6adbef08b5a520ebbd4638f),
the format Hugging Face model pages implement. This card describes **this repo's TTNN
port** of Meta's model, not the original release — see
[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) and Meta's own
[V-JEPA 2 paper](https://ai.meta.com/research/vjepa/) for the base model's own card.

## Model Details

- **Base model:** Meta's V-JEPA 2-AC — a self-supervised ViT-giant video encoder
  (40 blocks, 1408 hidden, 22 heads) plus an action-conditioned, frame-causal predictor
  (24 blocks, 1024 hidden, 16 heads), action-conditioned-post-trained on real robot
  video.
- **This artifact:** a from-scratch TTNN reimplementation of the encoder and predictor,
  ported layer-by-layer and checked against the reference PyTorch implementation at
  every stage (see [README.md](README.md)'s "Notable bring-up details").
- **Checkpoint:** `vjepa2-ac-vitg-fpc64-256-droid-tt` (a stripped, inference-only,
  bf16 copy of Meta's official checkpoint — same weights, smaller download; see
  `scripts/strip_checkpoint.py`). Also published at
  [`episod/vjepa2-ac-vitg-fpc64-256-droid-tt`](https://huggingface.co/episod/vjepa2-ac-vitg-fpc64-256-droid-tt).
- **Hardware target:** Tenstorrent Blackhole (single chip); a CPU reference backend
  (`--backend reference`) is also provided for portability/testing without hardware.
- **License:** MIT, matching the upstream `facebookresearch/vjepa2` license this is
  derived from.
- **Repo:** [tsingletaryTT/tt-vjepa2](https://github.com/tsingletaryTT/tt-vjepa2).

## Intended Use

- **Primary use:** a hardware bring-up and demonstration — proving V-JEPA2-AC's
  encoder/predictor run correctly and performantly on Blackhole, and serving as a
  building block (via the [ASGI service](docs/superpowers/specs/2026-09-08-asgi-service-design.md))
  for other things that want embedding-space video understanding or action-conditioned
  planning without reimplementing the port.
- **Supports:** zero-shot action-conditioned prediction (given a frame, state, and
  candidate action, predict the resulting embedding) and CEM-based planning (search
  over candidate actions toward a goal embedding) — the same capabilities Meta's own
  paper demonstrates, not new ones added by this port.
- **Out of scope:**
  - Pixel/video generation — this model never renders anything; every visual in the
    demo app is either a real frame, a real correctness check, or a clearly labeled
    imagined embedding-space rollout.
  - Production robot deployment without further validation — this repo's own testing
    is a single example clip and ad hoc demo checks, not a safety or reliability
    evaluation (see Evaluation Data and Caveats below).
  - Capabilities that belong to descendant models built on the same V-JEPA2 foundation
    but not implemented here: direct learned action policies (VLA-JEPA's approach),
    or language grounding (VL-JEPA's text decoder).

## Factors

- **Input:** one RGB frame (256×256) + a 7-DoF action/state delta (translation, Euler
  rotation, gripper) per step.
- **Precision:** bf16 for Linear layers (where the FLOPs are) and fp32 for LayerNorm
  and the residual stream (where 24–40-layer depth makes precision loss compound) —
  see README for the ablation that motivated this split.
- **Context length:** grows by one frame per planning/rollout step; behavior at very
  long rollouts is bounded by the rope/mask cache size in
  `tt/functional_predictor.py` (documented there, not yet addressed by a bucketing
  scheme — see Caveats).

## Metrics

- **Correctness:** Pearson correlation coefficient (PCC) against the unmodified
  reference PyTorch implementation, same real checkpoint weights, fp32.
- **Performance:** traced-replay latency (ms/forward) and throughput (frames/s) on a
  single Blackhole chip, compared to the same shape run through the unmodified
  reference implementation on CPU (see Quantitative Analyses).
- **Planning quality (this repo's own instrumentation, not a benchmark):** L2 distance
  in embedding space between a predicted and an actual real frame; CEM convergence
  curves toward a goal embedding.
- **IntPhys 2 (`scripts/eval_intphys2.py`):** pairwise accuracy — is the impossible
  video's mean L1 prediction "surprise" higher than its matched possible video's —
  the same violation-of-expectation methodology
  `facebookresearch/jepa-intuitive-physics` uses to evaluate V-JEPA models, adapted to
  the action-conditioned predictor this repo has (see Quantitative Analyses for the
  adaptation and its caveat).

## Evaluation Data

- **Correctness tests** (`tt/test_*.py`): synthetic random inputs run through real
  checkpoint weights, component-by-component (encoder alone, predictor alone, full
  pipeline).
- **Demo/Grounded Check:** a single real two-frame Franka arm clip and its recorded
  action, taken from Meta's own `energy_landscape_example.ipynb` notebook assets
  (`franka_example_traj.npz`) — one clip, not a benchmark suite.
- **IntPhys 2** ([`facebook/IntPhys2`](https://huggingface.co/datasets/facebook/IntPhys2)):
  the public `Main` eval split (1,012 videos, 506 possible/impossible pairs). The
  `HeldOut` split's ground truth is private (leaderboard-only), so this is a real,
  methodologically-standard number on the public data — not an official leaderboard
  submission.
- **Not evaluated by this repo:** MVPBench and CausalVQA — both require a genuine
  video-question-answering interface (text answers to questions about a video), which
  needs a language decoder this repo doesn't have (that's what VL-JEPA/VLA-JEPA add on
  top of a V-JEPA2-family encoder — see the descendant-model research this project did
  before choosing IntPhys 2). Also not evaluated: Something-Something v2,
  Epic-Kitchens-100, or a real-robot/LIBERO/SimplerEnv success-rate trial — named here
  rather than silently dropped, so the gap stays visible.

## Training Data

Not applicable — this repo performs no training. It loads Meta's own pretrained and
action-conditioned-post-trained checkpoint as-is. Per Meta's paper, the action-
conditioning post-training used under 62 hours of unlabeled robot video from the
DROID dataset; the base encoder was pretrained on over one million hours of internet
video. Neither training corpus is present in or used by this repo.

## Quantitative Analyses

Correctness (PCC against the reference implementation, fp32, real checkpoint weights):

| | PCC | bar |
|---|---|---|
| encoder (40 blocks) | 0.9970 | ≥ 0.995 |
| predictor (24 blocks) | 0.9972 | ≥ 0.995 |

Performance (8 frames @ 256px → 1024 context tokens → predictor, traced-replay,
single Blackhole chip):

| | latency | throughput | relative |
|---|---|---|---|
| Blackhole (TTNN, bf16-mixed, traced-replay) | 120.7 ms/forward | 66.3 input-frames/s | 1x |
| same machine's CPU, reference PyTorch (fp32 eager) | 4293.2 ms/forward | 1.86 input-frames/s | ~35.6x slower |

**IntPhys 2** (public `Main` split, 506 possible/impossible pairs, TTNN backend,
`scripts/results/intphys2_main_ttnn.json` has the raw per-video surprise scores):

| | pairwise accuracy | n pairs |
|---|---|---|
| Overall | **0.613** | 506 |
| by condition — permanence | 0.667 | |
| by condition — continuity | 0.692 | |
| by condition — immutability | 0.608 | |
| by condition — solidity | 0.507 (chance) | |
| by difficulty — Easy | 0.731 | |
| by difficulty — Medium | 0.605 | |
| by difficulty — Hard | 0.583 | |

Chance is 0.5. At n=506, 0.613 is ~5 standard deviations above chance (binomial SD
≈0.022) — not noise. The difficulty gradient (Easy > Medium > Hard) is the more
convincing signal: a model with no real violation-of-expectation sensitivity would not
reliably degrade with difficulty. **The zero-action proxy stated in Evaluation Data is
a real methodological compromise, but empirically it isn't erasing the signal** — worth
knowing precisely because an early 60-video debug run (`intphys2_debug_reference.json`)
came back at exactly chance (0.500, n=30) and looked like it might indicate the proxy
doesn't work at all; the full run showed that was an underpowered sample (binomial SD
≈0.09 at n=30), not a real finding. One honest exception: **solidity is at chance** in
the full run too — this port shows no measurable violation-of-expectation sensitivity
for that specific physical property, stated plainly rather than averaged away by the
overall number.

**Still missing, explicitly:** MVPBench, CausalVQA, Something-Something v2,
Epic-Kitchens-100, and any real-robot/LIBERO/SimplerEnv success-rate trial. The
Grounded Check and CEM convergence numbers surfaced in the demo app remain real but
demo-quality evidence, not benchmark-comparable claims, for whatever this port's
abilities aren't covered by IntPhys 2 above.

## Ethical Considerations

- This is a world model, not a generative video model — it cannot be used to
  synthesize fake video or images; every output is an embedding, and the demo app is
  explicit about labeling real vs. imagined content for exactly this reason.
- CEM search has no built-in collision or safety awareness beyond whatever the base
  model implicitly learned from training data — any robot deployment using this
  planner needs independent safety supervision, not just a low predicted-error score.

## Caveats and Recommendations

- The demo and correctness checks in this repo revolve around a single real example
  clip — not representative of task or environment diversity. Treat every number in
  this card as "this port behaves correctly and quickly on this one input shape and
  this one clip," not as a generalization claim.
- CEM search (`planning.cem_search`, `plan_step`) is stochastic and unseeded by
  design — the same inputs can produce different found actions across runs. This is
  a deliberate choice (see `docs/superpowers/specs/2026-09-08-asgi-service-design.md`
  and the Dance tab's "Plan with CEM" toggle discussion), not a bug, but it does mean
  results are not bit-reproducible run to run.
- The reference implementation's RoPE frequency bug (block-duplicated rather than
  interleaved) is reproduced on purpose, not fixed — see README's "Notable bring-up
  details." This port is bit-faithful to the pretrained weights' expectations, not to
  the "correct" RoPE convention.
- Long imagination rollouts (many chained planning steps) are bounded by a small
  rope/mask cache in `tt/functional_predictor.py`; very long rollouts beyond the
  cached bucket sizes fall back to full recompiles per new length, which is slow but
  not incorrect.
