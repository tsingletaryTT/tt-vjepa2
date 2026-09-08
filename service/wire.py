# SPDX-License-Identifier: MIT
"""Wire (de)serialization for the ASGI service: tensors as base64-encoded safetensors
(not pickle -- no arbitrary code execution risk, and it's the standard for exactly this),
frames as base64-encoded PNG (lossless round-trip for uint8 HxWx3 arrays)."""

import base64
import io

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

TENSOR_NAME = "t"


def tensor_to_b64(tensor: torch.Tensor) -> str:
    raw = safetensors_save({TENSOR_NAME: tensor.contiguous()})
    return base64.b64encode(raw).decode("ascii")


def tensor_from_b64(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    return safetensors_load(raw)[TENSOR_NAME]


def frame_to_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def frame_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(raw)))
