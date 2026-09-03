#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strip an official V-JEPA2-AC checkpoint down to what inference needs.

The official checkpoint (dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt, ~11GB) is a
full training checkpoint: encoder + predictor weights, optimizer state, a grad scaler,
and a momentum target-encoder copy of the weights. None of the training-only state is
used by `tt.functional_encoder`/`tt.functional_predictor` at inference time, and the
weights themselves only need to be fp32 on disk because training needed the precision --
this port casts them to bf16 on-device anyway (see functional_encoder.py's mixed-
precision notes), so storing them as fp32 buys nothing.

This script keeps only the encoder and predictor state dicts, casts every tensor to
bf16, and drops everything else. Same weights, ~4.2x smaller file
(11GB -> ~2.6GB) -- not a different or fine-tuned model, just a smaller download of the
one linked below.

Usage:
    python scripts/strip_checkpoint.py \\
        --input /path/to/vjepa2-ac-vitg.pt \\
        --output /path/to/vjepa2-ac-vitg.inference.pt

Download the official checkpoint first:
    wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt

Or skip this step entirely and use the pre-stripped checkpoint already published at
https://huggingface.co/episod/vjepa2-ac-vitg-fpc64-256-droid-tt
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="path to the official vjepa2-ac-vitg.pt checkpoint")
    parser.add_argument("--output", required=True, help="path to write the stripped checkpoint to")
    args = parser.parse_args()

    print(f"Loading {args.input} (this is the full ~11GB training checkpoint, may take a while)...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    stripped = {
        "encoder": {k: v.to(torch.bfloat16) if torch.is_tensor(v) else v for k, v in ckpt["encoder"].items()},
        "predictor": {k: v.to(torch.bfloat16) if torch.is_tensor(v) else v for k, v in ckpt["predictor"].items()},
    }

    print(f"Writing {args.output}...")
    torch.save(stripped, args.output)
    print("Done. Dropped: optimizer state, grad scaler, target_encoder. Cast: fp32 -> bf16.")


if __name__ == "__main__":
    main()
