"""CPU baseline for the same forward-pass shape as tt/benchmark.py, using the unmodified
reference PyTorch implementation (facebookresearch/vjepa2) -- same weights, same shape,
same warmup/timed-iteration methodology, only the hardware/framework differ. Run on the
host CPU (no GPU available when this baseline was measured).

Requires: `git clone https://github.com/facebookresearch/vjepa2 reference` at the repo
root. No TT_METAL_HOME/ttnn needed -- this is pure PyTorch, nothing device-specific."""

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "reference"))

from src.models.vision_transformer import VisionTransformer  # noqa: E402
from src.models.ac_predictor import VisionTransformerPredictorAC  # noqa: E402

CKPT_PATH = "/home/ttuser/.cache/vjepa2/vjepa2-ac-vitg.inference.pt"
WARMUP_ITERS = 2
TIMED_ITERS = 5


def main():
    torch.set_num_threads(torch.get_num_threads())  # use all available CPU threads (default)
    print(f"torch CPU threads: {torch.get_num_threads()}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    enc_sd = {k: v.float() for k, v in ckpt["encoder"].items()}
    pred_sd = {k: v.float() for k, v in ckpt["predictor"].items()}

    B, C, T, Hpx, Wpx = 1, 3, 8, 256, 256
    patch_size, tubelet_size = 16, 2
    gT, gH, gW = T // tubelet_size, Hpx // patch_size, Wpx // patch_size
    HW = gH * gW

    torch.manual_seed(0)
    pixel_values = torch.randn(B, C, T, Hpx, Wpx)
    actions = torch.randn(B, gT, 7)
    states = torch.randn(B, gT, 7)

    encoder = VisionTransformer(
        img_size=Hpx, patch_size=patch_size, num_frames=T, tubelet_size=tubelet_size,
        in_chans=C, embed_dim=1408, depth=40, num_heads=22, mlp_ratio=4.363636363636363,
        qkv_bias=True, use_rope=True, use_sdpa=True,
    )
    encoder.eval()
    encoder.load_state_dict({k[len("module.") :]: v for k, v in enc_sd.items() if k.startswith("module.")}, strict=False)

    predictor = VisionTransformerPredictorAC(
        img_size=Hpx, patch_size=patch_size, num_frames=T, tubelet_size=tubelet_size,
        embed_dim=1408, predictor_embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
        qkv_bias=True, is_frame_causal=True, use_rope=True, action_embed_dim=7, use_extrinsics=False,
    )
    predictor.eval()
    predictor.load_state_dict({k[len("module.") :]: v for k, v in pred_sd.items() if k.startswith("module.")}, strict=False)

    def forward():
        with torch.no_grad():
            context = encoder(pixel_values)
            if isinstance(context, (list, tuple)):
                context = context[-1]
            return predictor(context, actions, states)

    print(f"Warming up ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
        forward()

    print(f"Timing {TIMED_ITERS} iterations...")
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        out = forward()
    t1 = time.perf_counter()

    total_s = t1 - t0
    per_iter_ms = (total_s / TIMED_ITERS) * 1000
    frames_per_s = (B * T * TIMED_ITERS) / total_s

    print(f"\noutput shape {tuple(out.shape)}")
    print("=" * 60)
    print(f"Shape: encoder input (B={B}, C={C}, T={T}, H={Hpx}, W={Wpx}) -> {gT*HW} tokens -> predictor")
    print(f"CPU latency: {per_iter_ms:.2f} ms/forward (encoder+predictor combined)")
    print(f"CPU throughput: {frames_per_s:.2f} input-frames/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
