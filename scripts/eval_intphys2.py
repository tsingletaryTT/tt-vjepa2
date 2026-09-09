# SPDX-License-Identifier: MIT
"""Runs the IntPhys 2 violation-of-expectation evaluation
(https://huggingface.co/datasets/facebook/IntPhys2) against this repo's V-JEPA2-AC
port. Adapted from the official methodology
(facebookresearch/jepa-intuitive-physics's evals/intphys_test/eval.py): L1 loss
between predicted and actual layer-normed embeddings, aggregated per video into a
"surprise" score, then pairwise accuracy -- is the impossible video's surprise higher
than its matched possible video's.

Adaptation, stated plainly: the official eval uses V-JEPA's base masked-prediction
predictor (context patches -> masked future patches, one forward pass). This repo
only has the ACTION-CONDITIONED predictor (trained for robot manipulation), so this
instead runs an autoregressive, zero-action rollout -- the same "Grounded Check"
methodology already in gradio_app/app.py, extended across more than two frames. Zero
action is a defensible proxy for "no intervention" on these passively-observed
physics clips, not an exact match to how Meta evaluated their base model on this
benchmark. See MODEL_CARD.md for this caveat stated where the results are reported.

Usage:
    python scripts/eval_intphys2.py --backend reference --split debug
    python scripts/eval_intphys2.py --backend ttnn --split main --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradio_app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from intphys2_scoring import accuracy_by_group, load_metadata, pair_videos, pairwise_accuracy  # noqa: E402

N_FRAMES = 8  # matches this repo's own benchmark.py convention (8 frames)
IMG_SIZE = 256


def sample_frames(video_path: str, n_frames: int = N_FRAMES) -> np.ndarray:
    """Uniformly samples n_frames across the clip's real duration, resized to what
    the encoder expects. Reads sequentially (not by seeking) since seek accuracy on
    compressed video depends on keyframe placement -- sequential read is slower but
    exact regardless of codec."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, total - 1, n_frames).astype(int).tolist())

    captured = {}
    i = 0
    while len(captured) < len(wanted):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if i in wanted:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            captured[i] = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        i += 1
    cap.release()

    ordered_indices = sorted(captured)
    return np.stack([captured[i] for i in ordered_indices])  # [n_frames, H, W, 3] uint8


def l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)))


def video_surprise(backend, frames: np.ndarray) -> float:
    """Encode frame 0, then autoregressively predict frames 1..N-1 conditioned on a
    ZERO action/state at every step, comparing each prediction to that frame's own
    real encoding. Mean L1 error across steps is the video's surprise score -- the
    same chaining imagination_rollout (gradio_app/app.py) already does, driven by
    zero actions instead of a real or scripted action sequence."""
    n = len(frames)
    rep0 = backend.encode_frame(frames[0])
    reps = rep0.unsqueeze(1)  # [1,1,HW,D]
    zero_actions = np.zeros((n - 1, 7), dtype=np.float32)
    zero_states = np.zeros((n - 1, 7), dtype=np.float32)

    losses = []
    for i in range(1, n):
        actions_t = torch.from_numpy(zero_actions[:i]).float().unsqueeze(0)
        states_t = torch.from_numpy(zero_states[:i]).float().unsqueeze(0)
        next_rep, _ = backend.predict_step(reps, actions_t, states_t)
        actual_rep = backend.encode_frame(frames[i])
        losses.append(l1(next_rep, actual_rep))
        reps = torch.cat([reps, next_rep.unsqueeze(1)], dim=1)
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ttnn", "reference"], default="reference")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--split", choices=["debug", "main"], default="debug")
    parser.add_argument("--out", default=None, help="Path to write raw per-video scores JSON")
    args = parser.parse_args()

    split_dir = "Debug" if args.split == "debug" else "Main"
    print(f"Downloading IntPhys2/{split_dir} (if not already cached)...")
    data_root = snapshot_download("facebook/IntPhys2", repo_type="dataset", allow_patterns=[f"{split_dir}/*"])
    csv_path = Path(data_root) / split_dir / "metadata.csv"
    records = load_metadata(str(csv_path))
    pairs = pair_videos(records)
    print(f"{len(records)} videos, {len(pairs)} possible/impossible pairs")

    if args.backend == "ttnn":
        from backends import TTNNBackend

        backend = TTNNBackend(device_id=args.device_id)
    else:
        from backends import ReferenceBackend

        backend = ReferenceBackend()

    scores: dict[str, float] = {}
    try:
        for i, r in enumerate(records):
            video_path = str(Path(data_root) / split_dir / r.file_name)
            frames = sample_frames(video_path)
            scores[r.name] = video_surprise(backend, frames)
            print(
                f"  [{i + 1}/{len(records)}] {r.name} ({r.condition}, {r.difficulty}): "
                f"surprise={scores[r.name]:.4f}"
            )
    finally:
        if hasattr(backend, "close"):
            backend.close()

    overall = pairwise_accuracy([(scores[p.name], scores[imp.name]) for p, imp in pairs])
    by_condition = accuracy_by_group(pairs, scores, lambda r: r.condition)
    by_difficulty = accuracy_by_group(pairs, scores, lambda r: r.difficulty)

    print(f"\n=== IntPhys2 ({args.split} split, backend={backend.name}) ===")
    print(f"Overall pairwise accuracy: {overall:.4f}  ({len(pairs)} pairs)")
    print(f"By condition: {by_condition}")
    print(f"By difficulty: {by_difficulty}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "scores": scores,
                    "overall_accuracy": overall,
                    "by_condition": by_condition,
                    "by_difficulty": by_difficulty,
                    "backend": backend.name,
                    "split": args.split,
                    "n_pairs": len(pairs),
                },
                f,
                indent=2,
            )
        print(f"Wrote raw results to {args.out}")


if __name__ == "__main__":
    main()
