# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Full 40-block encoder, TTNN vs the real facebookresearch/vjepa2 VisionTransformer,
real (stripped) checkpoint weights. Same small synthetic clip as the single-block check
(fast iteration); scaling to the full advertised 64-frame/384px clip is the next stage,
not this one -- this stage proves the block stack composes correctly end to end."""

import sys
from pathlib import Path

import torch

import ttnn

AUTOPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTOPORT / "reference"))

from src.models.vision_transformer import VisionTransformer  # noqa: E402

sys.path.insert(0, str(AUTOPORT.parent.parent.parent))
sys.path.insert(0, str(AUTOPORT))  # AUTOPORT/tt/ is the "tt" package itself

from tt.functional_encoder import VJEPA2Encoder, VJEPA2EncoderConfig  # noqa: E402
from tt.test_functional_encoder import CKPT_PATH, pcc  # noqa: E402


def main():
    # Real operating resolution for this checkpoint: the AC finetune's own training
    # config (configs/train/vitg16/droid-256px-8f.yaml) is 8 frames @ 256px -> 4 tubelets
    # x 16x16 patches = 1024 tokens. A tiny 64-token white-noise smoke test left the
    # 40-block PCC seed-sensitive right at the 0.995 boundary (0.993-0.997 depending on
    # seed) -- real resolution, closer to what the model was actually trained to see,
    # is the test that actually matters here, not a lucky seed on an adversarially small
    # synthetic clip. grid_size must match: 256 // patch_size = 16.
    cfg = VJEPA2EncoderConfig(grid_size=16)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["encoder"].items()}

    B, C, T, Hpx, Wpx = 1, cfg.in_chans, 8, 256, 256
    torch.manual_seed(0)
    pixel_values = torch.randn(B, C, T, Hpx, Wpx)

    ref = VisionTransformer(
        img_size=Hpx, patch_size=cfg.patch_size, num_frames=T, tubelet_size=cfg.tubelet_size,
        in_chans=cfg.in_chans, embed_dim=cfg.hidden_size, depth=cfg.num_layers, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, qkv_bias=True, use_rope=True, use_sdpa=True,
    )
    ref.eval()
    missing, unexpected = ref.load_state_dict(
        {k[len("module.") :]: v for k, v in sd.items() if k.startswith("module.")}, strict=False
    )
    # pos_embed buffers (if any) are the only thing expected to differ -- use_rope=True
    # means the reference doesn't use learned pos_embed, so this should be empty/harmless.
    print(f"ref load_state_dict: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        ref_out = ref(pixel_values)
    if isinstance(ref_out, (list, tuple)):
        ref_out = ref_out[-1]

    device = ttnn.open_device(device_id=0)
    try:
        tt_encoder = VJEPA2Encoder.from_state_dict(sd, cfg=cfg, device=device)
        tt_out = tt_encoder.forward(pixel_values)

        tt_out_torch = ttnn.to_torch(tt_out).reshape(ref_out.shape)
        full_pcc = pcc(tt_out_torch, ref_out)
        print(f"full encoder (40 blocks) output PCC: {full_pcc:.6f}")
        assert full_pcc >= 0.995, f"full encoder PCC {full_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
