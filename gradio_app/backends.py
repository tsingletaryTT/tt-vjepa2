# SPDX-License-Identifier: MIT
"""Two interchangeable model backends for the demo app: one that runs our TTNN port
on real Blackhole hardware (local use, needs a gozer lease held by the caller), and one
that runs the unmodified reference PyTorch implementation on CPU (what an HF Space
without Tenstorrent hardware would use). Same interface, so the app doesn't care which
one is driving it.

Both implement the imagination-rollout pattern from Meta's own
`reference/notebooks/utils/world_model_wrapper.py::WorldModel`: encode a real starting
frame once, then chain the predictor's own output back in as the "next frame" with no
further camera observation -- this is exactly how their CEM planner imagines candidate
futures, not something invented for this demo.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent  # facebook_vjepa2_vitg_fpc64_384/
CKPT_PATH = "/home/ttuser/.cache/vjepa2/vjepa2-ac-vitg.inference.pt"

PATCH_SIZE = 16
TUBELET_SIZE = 2
IMG_SIZE = 256
GRID = IMG_SIZE // PATCH_SIZE  # 16 -> 256 tokens/frame


def _load_checkpoint():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    return ckpt["encoder"], ckpt["predictor"]


def _normalize_frame(frame_uint8: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 -> [1,3,2,H,W] float clip, duplicated across the tubelet (a single
    still image treated as a 2-frame clip, matching WorldModel.encode's convention).
    ImageNet mean/std, same normalization the reference transform applies."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    x = torch.from_numpy(frame_uint8).float().permute(2, 0, 1) / 255.0  # [3,H,W]
    x = x.unsqueeze(0).unsqueeze(2).repeat(1, 1, TUBELET_SIZE, 1, 1)  # [1,3,2,H,W]
    return (x - mean) / std


class ReferenceBackend:
    """Pure PyTorch, CPU (or CUDA if available). What an HF Space runs."""

    name = "reference-pytorch"

    def __init__(self, device: str = "cpu", max_steps: int = 24):
        sys.path.insert(0, str(REPO_ROOT / "reference"))
        from src.models.ac_predictor import VisionTransformerPredictorAC
        from src.models.vision_transformer import VisionTransformer

        self.device = device
        enc_sd, pred_sd = _load_checkpoint()
        enc_sd = {k: v.float() for k, v in enc_sd.items()}
        pred_sd = {k: v.float() for k, v in pred_sd.items()}

        self.encoder = (
            VisionTransformer(
                img_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                num_frames=TUBELET_SIZE,
                tubelet_size=TUBELET_SIZE,
                in_chans=3,
                embed_dim=1408,
                depth=40,
                num_heads=22,
                mlp_ratio=4.363636363636363,
                qkv_bias=True,
                use_rope=True,
                use_sdpa=True,
            )
            .eval()
            .to(device)
        )
        self.encoder.load_state_dict(
            {k[len("module.") :]: v for k, v in enc_sd.items() if k.startswith("module.")}, strict=False
        )

        # `num_frames` here sizes the predictor's precomputed frame-causal attention
        # mask (sliced down to the actual sequence length at forward time, see
        # VisionTransformerPredictorAC.forward) -- it must cover the longest rollout
        # this backend will be asked to do, i.e. TUBELET_SIZE * max_steps frames, not
        # the single starting frame.
        self.predictor = (
            VisionTransformerPredictorAC(
                img_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                num_frames=TUBELET_SIZE * max_steps,
                tubelet_size=TUBELET_SIZE,
                embed_dim=1408,
                predictor_embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4.0,
                qkv_bias=True,
                is_frame_causal=True,
                use_rope=True,
                action_embed_dim=7,
                use_extrinsics=False,
            )
            .eval()
            .to(device)
        )
        self.predictor.load_state_dict(
            {k[len("module.") :]: v for k, v in pred_sd.items() if k.startswith("module.")}, strict=False
        )

    def encode_frame(self, frame_uint8: np.ndarray) -> torch.Tensor:
        clip = _normalize_frame(frame_uint8).to(self.device)
        with torch.no_grad():
            h = self.encoder(clip)  # [1, HW, D]
        return F.layer_norm(h, (h.size(-1),))

    def predict_step(self, reps: torch.Tensor, actions: torch.Tensor, states: torch.Tensor):
        """reps: [1,T,HW,D] growing context. actions/states: [1,T,7]. Returns
        (next_rep [1,HW,D] normalized, latency_ms)."""
        B, T, N_T, D = reps.shape
        flat = reps.reshape(B, T * N_T, D)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.predictor(flat, actions, states)
        latency_ms = (time.perf_counter() - t0) * 1000
        next_rep = out[:, -N_T:]
        return F.layer_norm(next_rep, (next_rep.size(-1),)), latency_ms


class TTNNBackend:
    """Our tt-metal/TTNN port, real Blackhole hardware. Caller must already hold a
    gozer lease and have TT_VISIBLE_DEVICES set -- this class does not acquire one."""

    name = "ttnn-blackhole"

    def __init__(self, tt_metal_home: str | None = None, device_id: int = 0):
        tt_metal_home = tt_metal_home or os.environ.get("TT_METAL_HOME")
        if not tt_metal_home:
            raise RuntimeError("Set TT_METAL_HOME (or pass tt_metal_home=) to a tt-metal checkout.")
        sys.path.insert(0, tt_metal_home)
        sys.path.insert(0, str(REPO_ROOT))

        from tt.functional_encoder import VJEPA2Encoder, VJEPA2EncoderConfig
        from tt.functional_predictor import VJEPA2Predictor, VJEPA2PredictorConfig

        import ttnn

        self.ttnn = ttnn
        self.device = ttnn.open_device(device_id=device_id)
        enc_sd, pred_sd = _load_checkpoint()
        enc_sd = {k: v.float() for k, v in enc_sd.items()}
        pred_sd = {k: v.float() for k, v in pred_sd.items()}

        self.enc_cfg = VJEPA2EncoderConfig(grid_size=GRID)
        self.pred_cfg = VJEPA2PredictorConfig(grid_size=GRID)
        self.encoder = VJEPA2Encoder.from_state_dict(enc_sd, cfg=self.enc_cfg, device=self.device)
        self.predictor = VJEPA2Predictor.from_state_dict(pred_sd, cfg=self.pred_cfg, device=self.device)

    def close(self):
        self.ttnn.close_device(self.device)

    def encode_frame(self, frame_uint8: np.ndarray) -> torch.Tensor:
        clip = _normalize_frame(frame_uint8)  # [1,3,2,H,W]
        out = self.encoder.forward(clip)  # ttnn tensor
        h = self.ttnn.to_torch(out).reshape(1, GRID * GRID, self.enc_cfg.hidden_size)
        return F.layer_norm(h, (h.size(-1),))

    def predict_step(self, reps: torch.Tensor, actions: torch.Tensor, states: torch.Tensor):
        B, T, N_T, D = reps.shape
        flat = reps.reshape(B, T * N_T, D)
        context_tt = self.ttnn.from_torch(
            flat, dtype=self.ttnn.float32, layout=self.ttnn.TILE_LAYOUT, device=self.device
        )
        t0 = time.perf_counter()
        out_tt = self.predictor.forward(context_tt, actions, states, T, GRID, GRID)
        self.ttnn.synchronize_device(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000
        out = self.ttnn.to_torch(out_tt).reshape(B, T, N_T, D)
        next_rep = out[:, -1]
        return F.layer_norm(next_rep, (next_rep.size(-1),)), latency_ms
