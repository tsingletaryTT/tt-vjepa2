# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Traced-replay latency for the encoder -> predictor pipeline, at the AC checkpoint's
real operating shape (8 frames @ 256px -> 1024 tokens; single 1x1 device).

Per ttm-functional-decoder's tracing guidance: warmup/JIT-compile eagerly first, then
capture ONE forward pass as a trace (`ttnn.begin_trace_capture`/`end_trace_capture`),
then replay it with `ttnn.execute_trace` for the timed loop. This removes host dispatch
overhead between ops from the measured window -- what's left is device execution time.

Stable buffer addresses across replay are the actual constraint tracing imposes: pixel/
action/state inputs and the RoPE tables/attention mask are all prepared ONCE, before
capture, via the `prepare_input`/`prepare_conditioning`/`get_rope_tables`/
`get_rope_and_mask` methods added to VJEPA2Encoder/VJEPA2Predictor for exactly this --
`forward_device` on both classes is the pure device-op graph with no fresh allocations,
safe to capture.
"""

import os
import sys
import time
from pathlib import Path

import torch

import ttnn

# Requires TT_METAL_HOME set to a tt-metal checkout (for `ttnn` + `models.common.*`).
REPO_ROOT = Path(__file__).resolve().parent.parent
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout before running this script.")
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, str(REPO_ROOT))  # REPO_ROOT/tt/ is the "tt" package itself

from tt.functional_encoder import VJEPA2Encoder, VJEPA2EncoderConfig  # noqa: E402
from tt.functional_predictor import VJEPA2Predictor, VJEPA2PredictorConfig  # noqa: E402

CKPT_PATH = "/home/ttuser/.cache/vjepa2/vjepa2-ac-vitg.inference.pt"

WARMUP_ITERS = 3
TIMED_ITERS = 20


def main():
    enc_cfg = VJEPA2EncoderConfig(grid_size=16)
    pred_cfg = VJEPA2PredictorConfig(grid_size=16)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    enc_sd = {k: v.float() for k, v in ckpt["encoder"].items()}
    pred_sd = {k: v.float() for k, v in ckpt["predictor"].items()}

    B, C, T, Hpx, Wpx = 1, enc_cfg.in_chans, 8, 256, 256
    gT, gH, gW = T // enc_cfg.tubelet_size, Hpx // enc_cfg.patch_size, Wpx // enc_cfg.patch_size
    HW = gH * gW

    torch.manual_seed(0)
    pixel_values = torch.randn(B, C, T, Hpx, Wpx)
    actions = torch.randn(B, gT, pred_cfg.action_embed_dim)
    states = torch.randn(B, gT, pred_cfg.action_embed_dim)

    device = ttnn.open_device(device_id=0, trace_region_size=0)  # 0 = let ttnn auto-size it
    try:
        print("Building encoder (40 blocks) + predictor (24 blocks) from real weights...")
        encoder = VJEPA2Encoder.from_state_dict(enc_sd, cfg=enc_cfg, device=device)
        predictor = VJEPA2Predictor.from_state_dict(pred_sd, cfg=pred_cfg, device=device)

        print("Preparing persistent input buffers (one-time host->device transfer)...")
        patches_tt, grid = encoder.prepare_input(pixel_values)
        a_tt, s_tt = predictor.prepare_conditioning(actions, states)

        def forward():
            tokens = encoder.forward_device(patches_tt, grid, B)
            return predictor.forward_device(tokens, a_tt, s_tt, gT, gH, gW, B)

        print(f"Warming up eagerly ({WARMUP_ITERS} iters: JIT kernel build, populate rope/mask caches)...")
        for _ in range(WARMUP_ITERS):
            forward()
        ttnn.synchronize_device(device)

        print("Capturing trace (one forward pass, recorded not timed)...")
        trace_id = ttnn.begin_trace_capture(device)
        out = forward()
        ttnn.end_trace_capture(device, trace_id)

        print(f"Replaying trace for {TIMED_ITERS} timed iterations...")
        t0 = time.perf_counter()
        for _ in range(TIMED_ITERS):
            ttnn.execute_trace(device, trace_id, blocking=True)
        t1 = time.perf_counter()
        ttnn.release_trace(device, trace_id)

        total_s = t1 - t0
        per_iter_ms = (total_s / TIMED_ITERS) * 1000
        frames_per_s = (B * T * TIMED_ITERS) / total_s

        # sanity: output is non-garbage after replay
        out_torch = ttnn.to_torch(out)
        print(f"\noutput shape {tuple(out_torch.shape)}, mean={out_torch.mean().item():.4f}, "
              f"std={out_torch.std().item():.4f} (sanity check, not a correctness claim)")

        print()
        print("=" * 60)
        print(f"Shape: encoder input (B={B}, C={C}, T={T}, H={Hpx}, W={Wpx})")
        print(f"       -> {gT * HW} context tokens -> predictor -> {gT * HW} predicted tokens")
        print(f"Traced-replay latency: {per_iter_ms:.2f} ms/forward (encoder+predictor combined)")
        print(f"Throughput: {frames_per_s:.2f} input-frames/s")
        print("=" * 60)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
