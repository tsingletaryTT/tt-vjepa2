# SPDX-License-Identifier: MIT
"""Round-trip tests for the service's wire (de)serialization helpers -- tensors via
base64-encoded safetensors, frames via base64-encoded PNG. No FastAPI/hardware needed."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wire import frame_from_b64, frame_to_b64, tensor_from_b64, tensor_to_b64  # noqa: E402


def test_tensor_roundtrip_preserves_shape_and_values():
    t = torch.randn(3, 4, 5)
    decoded = tensor_from_b64(tensor_to_b64(t))
    assert decoded.shape == t.shape
    assert torch.allclose(decoded, t)


def test_tensor_roundtrip_preserves_dtype():
    t = torch.randn(2, 2).to(torch.bfloat16)
    decoded = tensor_from_b64(tensor_to_b64(t))
    assert decoded.dtype == torch.bfloat16


def test_frame_roundtrip_is_lossless():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    decoded = frame_from_b64(frame_to_b64(frame))
    assert decoded.shape == frame.shape
    assert decoded.dtype == frame.dtype
    np.testing.assert_array_equal(decoded, frame)
