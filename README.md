# tt-vjepa2

A [TT-Metal](https://github.com/tenstorrent/tt-metal) / TTNN bring-up of Meta's
[V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2): a self-supervised video
encoder plus an action-conditioned, frame-causal predictor that solves robot
manipulation tasks without task-specific training. Runs the ViT-giant encoder
(40 blocks, 1408 hidden, 22 heads) and the AC predictor (24 blocks, 1024 hidden, 16
heads) end-to-end on Tenstorrent Blackhole hardware.

For local development this depends on `ttnn` (a
[tt-metal](https://github.com/tenstorrent/tt-metal) checkout via `$TT_METAL_HOME`, or the
`ttnn` wheel) and on two small tooling modules from
[tt-transformers](https://github.com/tenstorrent/tt-transformers):
`tt_transformers.modules.lightweightmodule` and `tt_transformers.tensor_utils`. Put
tt-transformers' `src/` on `PYTHONPATH` (or install it) when you run the scripts below.
The served path ships as a v6 thin bundle that needs none of that on the host; see
"v6 thin packaging".

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, evaluation data, and — stated
plainly rather than glossed over — real results on the IntPhys 2 physical-reasoning
benchmark (61.3% pairwise accuracy, 506 pairs, clearly above chance) alongside which of
this model family's other usual benchmarks (MVPBench, CausalVQA, real-robot success
rate) still have *not* been run against this port.

## Results

8 frames @ 256px → 1024 context tokens → predictor → 1024 predicted tokens, traced-replay
latency on a single Blackhole chip:

| | latency | throughput | relative |
|---|---|---|---|
| Blackhole (TTNN, bf16-mixed, traced-replay) | 120.7 ms/forward | 66.3 input-frames/s | 1x |
| same machine's CPU, reference PyTorch (AMD Ryzen 7 9700X, 8c/16t, fp32 eager) | 4293.2 ms/forward | 1.86 input-frames/s | ~35.6x slower |

The CPU number is the unmodified `facebookresearch/vjepa2` reference implementation run
on the same host, same weights, same input shape — not a different framework's
optimized inference path, and not a GPU (none was available to benchmark against). Take
it as "how much faster than not having a Blackhole," not as a claim about GPU-class
hardware.

Correctness (PCC against the reference `facebookresearch/vjepa2` implementation, fp32,
on real checkpoint weights):

| | PCC | bar |
|---|---|---|
| encoder (40 blocks) | 0.9970 | ≥ 0.995 |
| predictor (24 blocks) | 0.9972 | ≥ 0.995 |

## What's here

- `tt/functional_encoder.py` — patch embed, 3-axis (depth/height/width) RoPE attention
  block, full 40-block encoder
- `tt/functional_predictor.py` — action/state conditioning, frame-causal AC predictor
- `tt/test_functional_encoder.py`, `tt/test_full_encoder.py`, `tt/test_functional_predictor.py` —
  PCC correctness checks against the reference implementation
- `tt/benchmark.py` — traced-replay end-to-end latency/throughput on Blackhole
- `tt/cpu_benchmark.py` — the same shape/methodology run through the unmodified
  reference PyTorch implementation, for the CPU comparison above
- `tt/profile_run.py` — per-op device-time breakdown via Tracy/`tt-perf-report`
- `scripts/strip_checkpoint.py` — shrinks the official 11GB training checkpoint down to
  a ~2.6GB bf16 inference-only one (same weights, smaller download — see the script's
  docstring). A pre-stripped copy is already published at
  [`episod/vjepa2-ac-vitg-fpc64-256-droid-tt`](https://huggingface.co/episod/vjepa2-ac-vitg-fpc64-256-droid-tt)
  if you'd rather skip this step.

## Notable bring-up details

- **RoPE frequency bug, reproduced on purpose.** The reference implementation
  block-duplicates rotation frequencies (`cat([freq.sin(), freq.sin()])`) rather than
  interleaving them — not the usual RoPE convention, but the pretrained weights were
  trained against it, so this port reproduces it bit-for-bit rather than "fixing" it.
- **Fused RoPE via `ttnn.experimental.rotary_embedding_llama`.** Verified bit-for-bit
  equivalent (PCC 1.0) to a manual per-axis rotate-half implementation before adopting —
  this one op replaced a per-axis slice→rotate→concat loop and cut end-to-end latency by
  roughly 24x, since that loop's reshape/slice/permute scaffolding — not matmul — turned
  out to dominate device time.
- **Mixed precision.** bf16 weights and compute for every Linear layer (where the FLOPs
  are), fp32 for LayerNorm and the residual stream (where 40-64-layer-depth precision
  sensitivity lives). Empirically beats both pure-bf16 and pure-fp32.
- **Manual attention for the frame-causal predictor mask.** `ttnn.transformer.
  scaled_dot_product_attention`'s `attn_mask` doesn't produce correct results for this
  mask's block structure (isolated: masked SDPA gives PCC 0.927 vs. 0.997 unmasked,
  independent of mask magnitude — points at the kernel, not a numerics issue). A manual
  matmul→softmax→matmul with the same additive bias gives PCC 0.9999.

## Requirements

- A [tt-metal](https://github.com/tenstorrent/tt-metal) checkout with `ttnn` built —
  needed by everything except `cpu_benchmark.py` and the Gradio app in `--backend reference` mode
- A clone of the reference implementation at the repo root — needed by the correctness
  tests, `cpu_benchmark.py`, and the Gradio app, not by `benchmark.py`/`profile_run.py`
- Tenstorrent Blackhole hardware — needed by everything except `cpu_benchmark.py` and
  the Gradio app in `--backend reference` mode
- The stripped checkpoint (see above) at a path of your choosing

## Running

```bash
git clone https://github.com/facebookresearch/vjepa2 reference
export TT_METAL_HOME=/path/to/tt-metal   # not needed for cpu_benchmark.py or --backend reference
export PYTHONPATH=/path/to/tt-transformers/src:$PYTHONPATH   # the two tooling modules (see top)

mkdir -p ~/.cache/vjepa2
cp /path/to/vjepa2-ac-vitg.inference.pt ~/.cache/vjepa2/  # matches the CKPT_PATH each script hardcodes

python tt/test_functional_encoder.py
python tt/test_full_encoder.py
python tt/test_functional_predictor.py
python tt/benchmark.py
python tt/cpu_benchmark.py
```

## Gradio demo

```bash
pip install gradio plotly scipy
python gradio_app/app.py --backend ttnn        # real Blackhole hardware (default)
python gradio_app/app.py --backend reference   # CPU, no tt-metal/hardware needed
python gradio_app/app.py --backend remote --service-url http://127.0.0.1:8000  # ASGI service, see below
```

Three tabs, all clearly labeled real-vs-imagined (the model predicts embeddings, never
pixels or joint angles):

- **Grounded prediction check** — predicts the real second frame of Meta's own example
  clip from the first + the actual recorded action, compares to ground truth, and
  sweeps a small action grid (same methodology as Meta's own
  `energy_landscape_example.ipynb`) to show the real action sitting near the
  low-error point.
- **Make it dance** — chains named move primitives (`moves.py`: `WELLE`, `SPIN`,
  `VERBEUGUNG`, `SCHNAPP`, `ACHT`) into a choreography, played out via the same
  imagination-rollout chaining Meta's own CEM planner uses to evaluate candidate
  futures. A 2-link end-effector arm animates the result (see `robot_viz.py` for the
  tt-toplike-inspired color law and the forward-kinematics caveats).
- **CEM planning** — a real port of Meta's own Cross-Entropy Method optimizer
  (`planning.py`), iteratively searching for the action that best predicts the real
  frame 1 from frame 0, then reporting how close it converges to the action that was
  actually taken.

## ASGI service

A standalone FastAPI service exposing the encoder/predictor/planner over HTTP, so
other things can use this model without going through Gradio at all — Gradio itself
can run as a client of it (`--backend remote` above). It owns the device exclusively
and serializes every call onto a single dedicated worker, so any number of clients can
share it safely. See
[docs/superpowers/specs/2026-09-08-asgi-service-design.md](docs/superpowers/specs/2026-09-08-asgi-service-design.md)
for the full design and wire format.

```bash
pip install fastapi uvicorn httpx safetensors pillow
python service/main.py --backend ttnn        # real Blackhole hardware (default)
python service/main.py --backend reference   # CPU, no tt-metal/hardware needed
```

Endpoints: `POST /encode` (frame → embedding), `POST /predict_step` (mirrors the
in-process primitive exactly — the caller owns its own growing context), and
`POST /plan_step` (goal-directed: CEM searches for a real action reaching a probed
goal, wraps `planning.plan_step`).

## v6 thin packaging

Packaged for [tt-model-manager](https://github.com/tenstorrent/tt-model-manager)'s
`tt-dit-server` kind (no tokens, KV cache or continuous batching, so there is nothing for
vLLM to do), the same non-LLM kind tt-waypoint and tt-skyreels use. `service.main:app` is
the served ASGI target. Its module-level `app` touches no hardware at import time: routes
read `app.state.worker`, which is filled only when the `_lifespan` startup handler runs.

```bash
packaging/package-thin.sh <out-dir>     # stages only; never pushes
```

`packaging/package-thin.sh` is the one rerunnable recipe, and each file next to it has one
job:

| file | job |
|---|---|
| `package-thin.sh` | builds the inputs, then runs `tt-model package-thin` without a repo id |
| `requirements.txt`, `constraints.txt` | the bundle's pins; `regen-constraints.sh` rewrites the second from the first |
| `make_tooling_wheel.py` | cuts the tooling-only tt-transformers wheel from a pinned commit |
| `model.py` | the `--model-py` entry point (no vLLM adapter is needed for this kind) |
| `test_package_thin.py` | checks the recipe's flags, imports and install guards without a card |

The bundle follows the canonical v6 shape from tt-model-manager's `docs/E2E_RECIPE_V6.md`:

- **Engine:** stock `ttnn==0.78.0` from PyPI, the release this model was verified on.
- **Custom kernels:** none. The model is written in ttnn ops, so there is no `generic_op`
  wheel and `deps.custom_ops` is empty. This is the one part of the shape that does not apply.
- **Tooling:** tt-transformers as tooling only (a wheel with `models/` emptied and its own
  pins removed, shipped with `--ops-wheel`). The model uses two of its modules,
  `modules.lightweightmodule` and `tensor_utils.get_rot_transformation_mat`. The latter is
  identical to the copy in tt-metal v0.78.0 that the model used before.
- **Model code:** `tt/`, `gradio_app/` (backends and planning only) and `service/` ship inside
  the bundle as `extra_code/`, from an explicit file list in the script.
- **Dependency safety:** `--constraints` with every package pinned, and no index options in
  `requirements.txt`. `huggingface_hub` ships as a bundled wheel (sha256-checked) because its
  closure (`requests`, `certifi`, `urllib3`) is what the PyTorch CPU index would supply at
  stale versions. `idna` is a direct-URL requirement for the same reason.
- **Capabilities:** no tool or reasoning parser applies; this model has no tokens.
- **Weights:** the checkpoint is a pointer to
  `episod/vjepa2-ac-vitg-fpc64-256-droid-tt`, pinned to revision `16c9081`. The served code
  resolves it with `$VJEPA2_CKPT`, then the developer copy at `CKPT_PATH` if present, then
  `hf_hub_download` of that pinned revision. Before this change the served path read
  `CKPT_PATH` only, a path that exists on the bring-up machine and nowhere else.

**What is verified and what is not.** The recipe stages cleanly, `install.sh` finishes in a
copy of the staged folder with every `--verify` guard passing, and `import model` works from
that copy without importing `ttnn`. The earlier hardware results (a real `/encode`,
`/predict_step` and `/plan_step` on one Blackhole chip) were obtained with the previous
bundle shape. They have not been repeated on this one. Serving the new bundle on a card and
repeating those three calls is still to do.

**An earlier bug worth remembering.** The first packaged bundle's `/encode` call failed with
`No module named 'models.common.tensor_utils'`. `functional_encoder.py` imports it inside a
function body, so a grep of top-of-file imports missed it. `test_package_thin.py` now walks
every import in the shipped files, lazy ones included. A build-time check that never opens a
device cannot see this kind of fault, so serve the bundle once on hardware before publishing.

## IntPhys 2 evaluation

Runs the [IntPhys 2](https://huggingface.co/datasets/facebook/IntPhys2)
violation-of-expectation benchmark against this port. See
[MODEL_CARD.md](MODEL_CARD.md#quantitative-analyses) for the result and its
methodology/caveats, and `scripts/eval_intphys2.py`'s docstring for the full
adaptation notes.

```bash
pip install opencv-python-headless huggingface_hub
python scripts/eval_intphys2.py --backend reference --split debug   # ~60 videos, quick sanity check
python scripts/eval_intphys2.py --backend ttnn --split main --out results.json  # full 1,012-video public eval set
```

The dataset (1.82 GB, public `Main`/`Debug` splits) downloads automatically on first
run via `huggingface_hub`. Raw per-video surprise scores from the run behind the
Model Card's numbers are checked in at `scripts/results/`.

## License

MIT, matching the upstream [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2)
license this is derived from. See [LICENSE](LICENSE).
