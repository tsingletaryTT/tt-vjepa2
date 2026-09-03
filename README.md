# tt-vjepa2

A [TT-Metal](https://github.com/tenstorrent/tt-metal) / TTNN bring-up of Meta's
[V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2): a self-supervised video
encoder plus an action-conditioned, frame-causal predictor that solves robot
manipulation tasks without task-specific training. Runs the ViT-giant encoder
(40 blocks, 1408 hidden, 22 heads) and the AC predictor (24 blocks, 1024 hidden, 16
heads) end-to-end on Tenstorrent Blackhole hardware.

This repo is the bring-up code only — it depends on a
[tt-metal](https://github.com/tenstorrent/tt-metal) checkout for `ttnn` and a couple of
small shared utilities (`models.common.lightweightmodule`,
`models.common.tensor_utils`). It is not a standalone pip package.

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

## License

MIT, matching the upstream [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2)
license this is derived from. See [LICENSE](LICENSE).
