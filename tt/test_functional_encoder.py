# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""First correctness checkpoint: patch embed + one encoder block, TTNN vs the real
facebookresearch/vjepa2 reference, on real (stripped) checkpoint weights, small clip.

PCC >= 0.995 is the acceptance bar (ttm-functional-decoder's default), even though this
is an encoder not a decoder -- the bar itself doesn't depend on causal-vs-bidirectional.
"""

import sys
from pathlib import Path

import torch

import ttnn

AUTOPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTOPORT / "reference"))

from src.models.utils.modules import Block  # noqa: E402

sys.path.insert(0, str(AUTOPORT.parent.parent.parent))  # tt-metal-shaped root, for `models.*`
sys.path.insert(0, str(AUTOPORT))  # AUTOPORT/tt/ is the "tt" package itself

from tt.functional_encoder import (  # noqa: E402
    EncoderBlock,
    PatchEmbed3D,
    VJEPA2EncoderConfig,
    axis_positions_3d,
    build_fused_rope_table,
    get_rope_trans_mat,
)

CKPT_PATH = "/home/ttuser/.cache/vjepa2/vjepa2-ac-vitg.inference.pt"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    # grid_size must match the synthetic clip's own resolution (8x8 patches below), not
    # the checkpoint's real training resolution -- see VJEPA2EncoderConfig.grid_size docstring.
    cfg = VJEPA2EncoderConfig(grid_size=8)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ckpt["encoder"].items()}  # reference runs fp32

    # small synthetic clip: 1 tubelet (2 frames), 8x8 patches -> 64 tokens (tile-aligned,
    # SDPA requires seq_len a multiple of the 32x32 tile). Fast smoke test before scaling
    # to the real resolution/frame count.
    B, C, T, Hpx, Wpx = 1, cfg.in_chans, cfg.tubelet_size, 8 * cfg.patch_size, 8 * cfg.patch_size
    torch.manual_seed(0)
    pixel_values = torch.randn(B, C, T, Hpx, Wpx)

    # -- reference: patch embed (torch Conv3d) + one Block --
    ref_patch_embed = torch.nn.Conv3d(
        cfg.in_chans,
        cfg.hidden_size,
        kernel_size=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
        stride=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
    )
    ref_patch_embed.weight.data = sd["module.patch_embed.proj.weight"]
    ref_patch_embed.bias.data = sd["module.patch_embed.proj.bias"]
    ref_tokens = ref_patch_embed(pixel_values).flatten(2).transpose(1, 2)  # (B, N, C)

    ref_block = Block(
        dim=cfg.hidden_size,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        qkv_bias=True,
        use_rope=True,
        grid_size=cfg.grid_size,
        use_sdpa=True,
    )
    ref_block.eval()
    ref_block.load_state_dict(
        {k[len("module.blocks.0.") :]: v for k, v in sd.items() if k.startswith("module.blocks.0.")}
    )
    gT, gH, gW = T // cfg.tubelet_size, Hpx // cfg.patch_size, Wpx // cfg.patch_size
    with torch.no_grad():
        ref_out = ref_block(ref_tokens, T=gT, H_patches=gH, W_patches=gW)

    # -- TTNN --
    device = ttnn.open_device(device_id=0)
    try:
        tt_patch_embed = PatchEmbed3D.from_state_dict(sd, cfg=cfg, device=device)
        tt_block = EncoderBlock.from_state_dict(sd, layer_idx=0, cfg=cfg, device=device)

        tt_tokens, (gT2, gH2, gW2) = tt_patch_embed.forward(pixel_values)
        assert (gT2, gH2, gW2) == (gT, gH, gW)
        pos_d, pos_h, pos_w = axis_positions_3d(gT, gH, gW, cfg)
        cos_full, sin_full = build_fused_rope_table(pos_d, pos_h, pos_w, cfg, device, dtype=ttnn.bfloat16)
        trans_mat = get_rope_trans_mat(device, dtype=ttnn.bfloat16)
        rope_tables = (cos_full, sin_full, trans_mat)

        seq_len = gT * gH * gW
        tt_tokens_3d = ttnn.reshape(tt_tokens, (B, seq_len, cfg.hidden_size))
        tt_out = tt_block(tt_tokens_3d, rope_tables, B, seq_len)

        patch_embed_pcc = pcc(ttnn.to_torch(tt_tokens).reshape(ref_tokens.shape), ref_tokens)
        block_out_pcc = pcc(ttnn.to_torch(tt_out).reshape(ref_out.shape), ref_out)
        print(f"patch_embed PCC: {patch_embed_pcc:.6f}")
        print(f"block_0 output PCC: {block_out_pcc:.6f}")
        assert patch_embed_pcc >= 0.995, f"patch_embed PCC {patch_embed_pcc} < 0.995"
        assert block_out_pcc >= 0.995, f"block_0 PCC {block_out_pcc} < 0.995"
        print("PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
