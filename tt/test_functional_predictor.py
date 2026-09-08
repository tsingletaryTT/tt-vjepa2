# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Predictor correctness: TTNN vs the real facebookresearch/vjepa2 VisionTransformerPredictorAC,
real (stripped) checkpoint weights. Component-level test: synthetic random context tokens +
actions/states (not the encoder's actual output) -- isolates predictor correctness from
encoder correctness, per ttm-functional-decoder's "prefer a layer-only HF reference" guidance
generalized to "prefer a component-only reference before chaining the whole model"."""

import sys
from pathlib import Path

import torch

import ttnn

AUTOPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTOPORT / "reference"))
from src.models.ac_predictor import VisionTransformerPredictorAC  # noqa: E402

sys.path.insert(0, str(AUTOPORT.parent.parent.parent))
sys.path.insert(0, str(AUTOPORT))  # AUTOPORT/tt/ is the "tt" package itself
from tt.functional_predictor import VJEPA2Predictor, VJEPA2PredictorConfig  # noqa: E402
from tt.test_functional_encoder import CKPT_PATH, pcc  # noqa: E402


def main():
    cfg = VJEPA2PredictorConfig(grid_size=16)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["predictor"].items()}

    B, T, Hpx, Wpx, patch_size, tubelet_size = 1, 8, 256, 256, 16, 2
    gT, gH, gW = T // tubelet_size, Hpx // patch_size, Wpx // patch_size
    HW = gH * gW

    torch.manual_seed(0)
    context_tokens = torch.randn(B, gT * HW, cfg.encoder_hidden_size)
    actions = torch.randn(B, gT, cfg.action_embed_dim)
    states = torch.randn(B, gT, cfg.action_embed_dim)

    ref = VisionTransformerPredictorAC(
        img_size=Hpx,
        patch_size=patch_size,
        num_frames=T,
        tubelet_size=tubelet_size,
        embed_dim=cfg.encoder_hidden_size,
        predictor_embed_dim=cfg.pred_hidden_size,
        depth=cfg.pred_num_layers,
        num_heads=cfg.pred_num_heads,
        mlp_ratio=cfg.pred_mlp_ratio,
        qkv_bias=True,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=cfg.action_embed_dim,
        use_extrinsics=False,
    )
    ref.eval()
    missing, unexpected = ref.load_state_dict(
        {k[len("module.") :]: v for k, v in sd.items() if k.startswith("module.")}, strict=False
    )
    print(f"ref load_state_dict: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        ref_out = ref(context_tokens, actions, states)

    device = ttnn.open_device(device_id=0)
    try:
        tt_predictor = VJEPA2Predictor.from_state_dict(sd, cfg=cfg, device=device)
        context_tt = ttnn.from_torch(context_tokens, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_out = tt_predictor.forward(context_tt, actions, states, gT, gH, gW)

        tt_out_torch = ttnn.to_torch(tt_out).reshape(ref_out.shape)
        p = pcc(tt_out_torch, ref_out)
        print(f"predictor output PCC: {p:.6f}")
        assert p >= 0.995, f"predictor PCC {p} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
