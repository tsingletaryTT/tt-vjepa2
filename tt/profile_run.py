# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Per-op device profiling for the encoder -> predictor pipeline, real production shape.

Run under Tracy to get a per-op CSV, then post-process with tt-perf-report using the
PERF_FORWARD/PERF_FORWARD_END signposts to isolate the measured window from warmup:

    python -m tracy -r -v tt/profile_run.py
    tt-perf-report <generated ops_perf_results_*.csv> \
        --start-signpost PERF_FORWARD --end-signpost PERF_FORWARD_END

Eager (not traced) on purpose: tt-perf-report wants a per-op breakdown to answer "where
is the time going" (attention vs MLP vs the rope/reshape/concat scaffolding), and traced
replay collapses the graph into one opaque execution from the profiler's point of view.
"""

import os
import sys
from pathlib import Path

import torch

import ttnn

# Requires TT_METAL_HOME set to a tt-metal checkout (for `ttnn` + `models.common.*`).
REPO_ROOT = Path(__file__).resolve().parent.parent
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise RuntimeError("Set TT_METAL_HOME to a tt-metal checkout before running this script.")
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, str(REPO_ROOT))  # REPO_ROOT/tt/ is the package; REPO_ROOT itself must be on sys.path

from tt.functional_encoder import VJEPA2Encoder, VJEPA2EncoderConfig  # noqa: E402
from tt.functional_predictor import VJEPA2Predictor, VJEPA2PredictorConfig  # noqa: E402

CKPT_PATH = "/home/ttuser/.cache/vjepa2/vjepa2-ac-vitg.inference.pt"

WARMUP_ITERS = 1


def main():
    enc_cfg = VJEPA2EncoderConfig(grid_size=16)
    pred_cfg = VJEPA2PredictorConfig(grid_size=16)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    enc_sd = {k: v.float() for k, v in ckpt["encoder"].items()}
    pred_sd = {k: v.float() for k, v in ckpt["predictor"].items()}

    B, C, T, Hpx, Wpx = 1, enc_cfg.in_chans, 8, 256, 256
    gT, gH, gW = T // enc_cfg.tubelet_size, Hpx // enc_cfg.patch_size, Wpx // enc_cfg.patch_size

    torch.manual_seed(0)
    pixel_values = torch.randn(B, C, T, Hpx, Wpx)
    actions = torch.randn(B, gT, pred_cfg.action_embed_dim)
    states = torch.randn(B, gT, pred_cfg.action_embed_dim)

    device = ttnn.open_device(device_id=0)
    try:
        print("Building encoder (40 blocks) + predictor (24 blocks) from real weights...")
        encoder = VJEPA2Encoder.from_state_dict(enc_sd, cfg=enc_cfg, device=device)
        predictor = VJEPA2Predictor.from_state_dict(pred_sd, cfg=pred_cfg, device=device)

        patches_tt, grid = encoder.prepare_input(pixel_values)
        a_tt, s_tt = predictor.prepare_conditioning(actions, states)

        # Profiling ALL 64 blocks in one capture overflows the per-core profiler marker
        # buffer (1114+ ops from a single forward pass; Tracy warned markers were being
        # dropped and post-processing then crashed on the resulting gap). Every block is
        # structurally identical, so a representative slice (2 encoder + 2 predictor
        # blocks) gives the same per-op-type breakdown -- attention vs MLP vs the rope/
        # reshape/concat scaffolding -- without needing to capture all 64 repeats of it.
        N_BLOCKS_TO_PROFILE = 1

        def forward_sliced():
            cfg = encoder.cfg
            gTe, gHe, gWe = grid
            seq_len = gTe * gHe * gWe
            tokens = encoder.patch_embed.forward_device(patches_tt)
            x = ttnn.reshape(tokens, (B, seq_len, cfg.hidden_size))
            rope_tables = encoder.get_rope_tables(gTe, gHe, gWe)
            for block in encoder.blocks[:N_BLOCKS_TO_PROFILE]:
                x = block(x, rope_tables, B, seq_len)

            pred_cfg = predictor.cfg
            HW = gH * gW
            cond_tokens = pred_cfg.cond_tokens
            px = ttnn.linear(x, predictor.embed_w, bias=predictor.embed_b)
            px = ttnn.reshape(px, (B, gT, HW, pred_cfg.pred_hidden_size))
            a = ttnn.linear(a_tt, predictor.action_w, bias=predictor.action_b)
            s = ttnn.linear(s_tt, predictor.state_w, bias=predictor.state_b)
            px = ttnn.concat([a, s, px], dim=2)
            px = ttnn.reshape(px, (B, gT * (cond_tokens + HW), pred_cfg.pred_hidden_size))
            rope_tables, attn_mask = predictor.get_rope_and_mask(gT, gH, gW)
            for block in predictor.blocks[:N_BLOCKS_TO_PROFILE]:
                px = block(px, rope_tables, attn_mask, B, gT, HW)
            return px

        print(f"Warming up ({WARMUP_ITERS} iters: JIT kernel build, populate rope/mask caches)...")
        for _ in range(WARMUP_ITERS):
            forward_sliced()
        ttnn.synchronize_device(device)

        print(f"Signposted measured forward pass ({N_BLOCKS_TO_PROFILE} encoder + "
              f"{N_BLOCKS_TO_PROFILE} predictor blocks, representative slice)...")
        ttnn.profiler.tracy_message("PERF_FORWARD")
        out = forward_sliced()
        ttnn.synchronize_device(device)
        ttnn.profiler.tracy_message("PERF_FORWARD_END")

        out_torch = ttnn.to_torch(out)
        print(f"output shape {tuple(out_torch.shape)}, sanity mean={out_torch.mean().item():.4f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
