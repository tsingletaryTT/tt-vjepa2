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

| | latency | throughput |
|---|---|---|
| encoder + predictor, combined | 120.7 ms/forward | 66.3 input-frames/s |

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
- `tt/benchmark.py` — traced-replay end-to-end latency/throughput
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

- A [tt-metal](https://github.com/tenstorrent/tt-metal) checkout with `ttnn` built, on
  `PYTHONPATH` (or run from inside a tt-metal worktree)
- Tenstorrent Blackhole hardware
- The stripped checkpoint (see above) at a path of your choosing

## Running

Each script hardcodes `CKPT_PATH = "/home/ttuser/.cache/vjepa2/vjepa2-ac-vitg.inference.pt"`
at the top — either edit that constant or put your checkpoint at that path:

```bash
mkdir -p ~/.cache/vjepa2
cp /path/to/vjepa2-ac-vitg.inference.pt ~/.cache/vjepa2/

python tt/test_functional_encoder.py
python tt/test_full_encoder.py
python tt/test_functional_predictor.py
python tt/benchmark.py
```

## License

MIT, matching the upstream [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2)
license this is derived from. See [LICENSE](LICENSE).
