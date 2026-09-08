# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""TTNN functional bring-up of the facebook/vjepa2-vitg-fpc64-384 encoder.

This is a bidirectional ViT encoder, not a causal-LM decoder: no KV cache, no
prefill/decode split, one forward pass over every patch token at once. It borrows
`ttm-functional-decoder`'s correctness discipline (real reference forward pass,
PCC >= 0.995, single 1x1 mesh first, no hidden torch in the forward path) but not
its decoder-specific mechanics (paged cache, traced decode replay), which don't
apply to a non-autoregressive encoder.

Reference: models/autoports/facebook_vjepa2_vitg_fpc64_384/reference (facebookresearch/vjepa2,
src/models/utils/modules.py: Block + RoPEAttention, src/models/utils/patch_embed.py: PatchEmbed3D).
Checkpoint keys are Meta's own naming ("module.blocks.N....."), not HF `transformers`'
renamed vjepa2 port -- this file loads directly from that checkpoint format.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_USE_MANUAL_ATTENTION = False  # debug flag; see RoPEAttention.forward


def _hifi_compute_kernel_config():
    """HiFi4 + fp32 dest accumulation for every linear/layer_norm/attention call.

    NOTE on how this was actually debugged, because the first hypothesis was wrong: a
    40-block stack initially compounded to PCC 0.83 (single block: 0.9999). That gradual,
    non-cliff decline first looked like ordinary bf16 accumulation -- but neither this
    HiFi4 config nor fp32 storage moved the number at all (0.836 -> 0.837), which ruled
    that out. The real cause was a RoPE frequency-layout bug (see the block-duplicate
    comment in `_rope_axis_freqs`): a genuine, systematic per-layer error that just
    happened to look precision-shaped because it compounds with depth like one. Fixing
    that took 40-block PCC to ~0.988 (bf16 storage) / ~0.995-0.997 (fp32 storage,
    depending on input seed -- a tiny 64-token white-noise test sits close enough to the
    bar that seed variance alone crosses it). fp32 storage plus this compute config is
    kept as the correctness-proving default; bf16 weights are a `ttm-optimize` question
    for later, not a blocker here.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@dataclass
class VJEPA2EncoderConfig:
    hidden_size: int = 1408
    num_heads: int = 22
    num_layers: int = 40
    mlp_ratio: float = 4.363636363636363
    patch_size: int = 16
    tubelet_size: int = 2
    in_chans: int = 3
    layer_norm_eps: float = 1e-6
    # grid_size the reference RoPEAttention snaps h/w positions to. NOT a fixed per-
    # checkpoint constant -- the reference derives it as `img_size // patch_size` at
    # VisionTransformer construction time (vision_transformer.py: `grid_size=img_size[0]
    # // patch_size` in the Block instantiation loop). Caller must set this to match the
    # actual input resolution being run, or RoPE position-snapping silently diverges from
    # the reference for any resolution other than grid_size*patch_size square.
    grid_size: int = 16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def rope_axis_dim(self) -> int:
        # d_dim = h_dim = w_dim = 2 * ((head_dim // 3) // 2), per reference RoPEAttention.
        return 2 * ((self.head_dim // 3) // 2)


def _torch_linear_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    """HF/torch nn.Linear weight is (out, in); ttnn.linear wants (in, out)."""
    w = ttnn.from_torch(weight.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


def _torch_norm_to_ttnn(weight: torch.Tensor, bias: torch.Tensor, device, dtype=ttnn.float32):
    w = ttnn.from_torch(weight.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


class PatchEmbed3D(LightweightModule):
    """Non-overlapping Conv3d(stride == kernel) is exactly reshape-into-tubelets + Linear:
    no conv kernel needed. Each (tubelet_size, patch_size, patch_size) voxel becomes one
    flattened row multiplied by the conv weight reshaped to (embed_dim, in_chans*t*p*p)."""

    def __init__(
        self, weight_out_in: "ttnn.Tensor", bias: "ttnn.Tensor", cfg: VJEPA2EncoderConfig, device, dtype=ttnn.float32
    ):
        self.weight = weight_out_in  # already (in_features, embed_dim) for ttnn.linear
        self.bias = bias
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: VJEPA2EncoderConfig, device, dtype=ttnn.float32):
        # Weight is bf16 (mixed-precision default, matches EncoderBlock); `dtype` controls
        # the input activation only -- ttnn.linear accepts a higher-precision activation
        # against a bf16 weight without complaint, verified empirically (this is the very
        # first projection, worth keeping input-side fp32 for).
        w = state_dict["module.patch_embed.proj.weight"]
        embed_dim = w.shape[0]
        w = w.reshape(embed_dim, -1)  # (embed_dim, in_features)
        b = state_dict["module.patch_embed.proj.bias"]
        weight_out_in, bias = _torch_linear_to_ttnn(w, b, device, dtype=ttnn.bfloat16)
        return cls(weight_out_in, bias, cfg, device, dtype=dtype)

    def unfold_patches(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """Host-side only: unfold into non-overlapping (t, p, p) voxels, one flattened row
        per tubelet token, in the same (T, H, W) raster order Conv3d + flatten(2) produces.
        Split out so this (and the one-time device transfer that follows it) can happen
        ONCE, outside a captured trace -- trace capture only records device ops."""
        cfg = self.cfg
        B, C, T, H, W = pixel_values.shape
        t, p = cfg.tubelet_size, cfg.patch_size
        assert T % t == 0 and H % p == 0 and W % p == 0
        gT, gH, gW = T // t, H // p, W // p
        x = pixel_values.reshape(B, C, gT, t, gH, p, gW, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, gT, gH, gW, C, t, p, p)
        x = x.reshape(B, gT * gH * gW, C * t * p * p)
        return x, (gT, gH, gW)

    def to_device(self, patches: torch.Tensor) -> "ttnn.Tensor":
        """One-time host->device transfer of the unfolded patches. Call once; under trace
        capture/replay, only this tensor's CONTENTS should be refreshed between iterations
        (`ttnn.copy_host_to_device_tensor`), never re-allocated."""
        return ttnn.from_torch(patches, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def forward_device(self, x_tt: "ttnn.Tensor") -> "ttnn.Tensor":
        """Pure device op (the actual Conv3d-as-linear projection) -- safe to capture in a trace."""
        return ttnn.linear(x_tt, self.weight, bias=self.bias, compute_kernel_config=_hifi_compute_kernel_config())

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """Convenience path for correctness tests: does the host reshape + device transfer
        + projection every call. Not what a traced benchmark should use -- see
        `unfold_patches`/`to_device`/`forward_device` for the split version."""
        patches, grid = self.unfold_patches(pixel_values)
        x_tt = self.to_device(patches)
        return self.forward_device(x_tt), grid


def _rope_axis_freqs(pos: torch.Tensor, axis_dim: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for one RoPE axis, matching reference `rotate_queries_or_keys`
    exactly (including its documented duplicated-frequency quirk, kept for pretrained
    checkpoint compatibility)."""
    omega = torch.arange(axis_dim // 2, dtype=dtype)
    omega = omega / (axis_dim / 2.0)
    omega = 1.0 / (10000**omega)
    freq = torch.einsum("...n,f->...nf", pos.to(dtype), omega)  # (..., N, axis_dim/2)
    # Reference does `.squeeze(-1).repeat(1,1,1,2)` on a tensor whose last dim is
    # axis_dim/2 (not 1) -- squeeze is a no-op there, so repeat BLOCK-duplicates the whole
    # last axis: [f0..f9, f0..f9], not an interleave. That mismatches rotate_half's
    # consecutive-pair convention ((x0,x1) rotate together using f0,f1 respectively, two
    # DIFFERENT frequencies) -- a real bug, but the pretrained weights were trained against
    # it, so it must be reproduced bit-for-bit rather than "fixed". Block-duplicate via
    # cat, not repeat_interleave (which would give the interleaved, non-buggy version).
    emb_sin = torch.cat([freq.sin(), freq.sin()], dim=-1)
    emb_cos = torch.cat([freq.cos(), freq.cos()], dim=-1)
    return emb_cos, emb_sin


def axis_positions_3d(gT: int, gH: int, gW: int, cfg) -> tuple:
    """The three per-position axis values (frame/height/width) the encoder's tokens use,
    factored out so the predictor can build the same three arrays for its spatial tokens
    without duplicating this indexing math."""
    ids = torch.arange(gT * gH * gW)
    frame_ids = ids // (gH * gW)
    height_ids = (ids - frame_ids * gH * gW) // gW
    width_ids = (ids - frame_ids * gH * gW) - height_ids * gW
    return (
        frame_ids.float(),
        height_ids.float() * (cfg.grid_size / gH),
        width_ids.float() * (cfg.grid_size / gW),
    )


def build_fused_rope_table(
    pos_d: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    cfg,
    device,
    dtype=ttnn.bfloat16,
    identity_mask: "torch.Tensor | None" = None,
) -> "ttnn.Tensor":
    """cos/sin spanning the FULL head_dim in one tensor (d/h/w segments block-concatenated,
    any leftover head_dim padded with an identity rotation), for use with
    `ttnn.experimental.rotary_embedding_llama` -- one fused device call per q/k instead of
    three axis-slice-rotate-concat passes. Verified bit-for-bit equivalent (PCC 1.0 against
    the per-axis `_apply_axis_rope` loop this replaces) before adopting.

    `identity_mask`, if given, zeroes the h/w rotation (cos=1, sin=0) at the marked
    positions -- the predictor's cond (action/state) tokens, which the reference rotates
    only on the "d" axis and passes straight through otherwise; see
    functional_predictor.py's `build_unified_rope_tables` for why d can still be shared.
    """
    axis = cfg.rope_axis_dim
    head_dim = cfg.head_dim
    N = pos_d.shape[0]
    cos_d, sin_d = _rope_axis_freqs(pos_d.reshape(1, 1, -1), axis)
    cos_h, sin_h = _rope_axis_freqs(pos_h.reshape(1, 1, -1), axis)
    cos_w, sin_w = _rope_axis_freqs(pos_w.reshape(1, 1, -1), axis)
    if identity_mask is not None:
        cos_h[:, :, identity_mask, :] = 1.0
        sin_h[:, :, identity_mask, :] = 0.0
        cos_w[:, :, identity_mask, :] = 1.0
        sin_w[:, :, identity_mask, :] = 0.0
    remainder = head_dim - 3 * axis
    cos_full = torch.cat([cos_d, cos_h, cos_w, torch.ones(1, 1, N, remainder)], dim=-1)
    sin_full = torch.cat([sin_d, sin_h, sin_w, torch.zeros(1, 1, N, remainder)], dim=-1)
    return (
        ttnn.from_torch(cos_full, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(sin_full, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
    )


_ROPE_TRANS_MAT_CACHE = {}


def get_rope_trans_mat(device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    """The (1,1,32,32) adjacent-pair rotation matrix `rotary_embedding_llama` needs --
    constant, independent of position/resolution, so built once and reused for every
    call (q and k, every block, both the encoder and the predictor)."""
    from models.common.tensor_utils import get_rot_transformation_mat

    key = id(device)
    if key not in _ROPE_TRANS_MAT_CACHE:
        trans_mat = get_rot_transformation_mat(dhead=32)  # TILE_WIDTH; broadcasts per 32-col tile
        _ROPE_TRANS_MAT_CACHE[key] = ttnn.from_torch(trans_mat, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return _ROPE_TRANS_MAT_CACHE[key]


def _rotate_half(x: "ttnn.Tensor") -> "ttnn.Tensor":
    """Reference does: y = x.unflatten(-1,(-1,2)); y1,y2 = unbind; stack(-y2,y1); flatten.
    That interleaves pairs (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...), NOT the "rotate half the
    vector" convention common in LLM RoPE. Implemented as an even/odd split + restitch."""
    shape = tuple(x.shape)
    x = ttnn.reshape(x, (*shape[:-1], shape[-1] // 2, 2))
    x0 = x[..., 0]
    x1 = x[..., 1]
    rotated = ttnn.stack([ttnn.neg(x1), x0], dim=-1)
    return ttnn.reshape(rotated, shape)


def _apply_axis_rope(x_slice: "ttnn.Tensor", cos: "ttnn.Tensor", sin: "ttnn.Tensor") -> "ttnn.Tensor":
    return x_slice * cos + _rotate_half(x_slice) * sin


class RoPEAttention(LightweightModule):
    """Mirrors reference `RoPEAttention`: qkv -> split heads -> rotate q/k on the
    depth/height/width axis slices of head_dim (leaving any remainder unrotated) ->
    scaled-dot-product attention -> output proj. No mask (full bidirectional encoder,
    every token attends to every token)."""

    def __init__(self, qkv_w, qkv_b, proj_w, proj_b, cfg: VJEPA2EncoderConfig, device):
        self.qkv_w, self.qkv_b = qkv_w, qkv_b
        self.proj_w, self.proj_b = proj_w, proj_b
        self.cfg = cfg
        self.device = device

    @classmethod
    def from_state_dict(cls, state_dict, *, prefix: str, cfg: VJEPA2EncoderConfig, device, dtype=ttnn.float32):
        # Weights are always bf16 here regardless of `dtype` -- see the mixed-precision
        # note on EncoderBlock. `dtype` only controls the activation/residual side.
        qkv_w, qkv_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.attn.qkv.weight"], state_dict[f"{prefix}.attn.qkv.bias"], device, ttnn.bfloat16
        )
        proj_w, proj_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.attn.proj.weight"], state_dict[f"{prefix}.attn.proj.bias"], device, ttnn.bfloat16
        )
        return cls(qkv_w, qkv_b, proj_w, proj_b, cfg, device)

    def forward(self, x: "ttnn.Tensor", rope_tables: tuple, batch: int, seq_len: int) -> "ttnn.Tensor":
        """x arrives at whatever activation dtype the block uses (fp32 for the mixed-
        precision default); cast down to bf16 for the qkv/attention/proj compute (weights
        are bf16), then back up before returning, so the residual stream this feeds into
        stays high-precision."""
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        residual_dtype = x.dtype
        x = ttnn.typecast(x, ttnn.bfloat16)

        # Fused split-into-heads instead of reshape+slice+permute: profiling found ~95%
        # of device time going to exactly that kind of shape manipulation (reshape/
        # tilize/slice/permute), only ~2% to matmul. `nlp_create_qkv_heads` is tt-metal's
        # own primitive for this (models/common/modules/attention/attention_1d.py's
        # standard decoder pipeline uses it); verified bit-for-bit equivalent (max diff
        # 0.0156, ordinary bf16 rounding) against the manual reshape+permute it replaces.
        qkv = ttnn.linear(
            x, self.qkv_w, bias=self.qkv_b, compute_kernel_config=_hifi_compute_kernel_config()
        )  # (B, N, 3*H*D)
        qkv = ttnn.reshape(qkv, (batch, 1, seq_len, 3 * H * D))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, transpose_k_heads=False)

        # Fused rotary embedding: one device call per q/k instead of 3 axis-slice ->
        # rotate -> concat passes (each rotate itself being reshape+slice+stack+reshape).
        # `rope_tables` is (cos_full, sin_full, trans_mat) from build_fused_rope_table +
        # get_rope_trans_mat, spanning the whole head_dim -- verified PCC 1.0 against the
        # old per-axis loop before switching. See that function's docstring for why one
        # fused call reproduces all three axis rotations (plus the untouched remainder)
        # at once.
        cos_full, sin_full, trans_mat = rope_tables
        q = ttnn.experimental.rotary_embedding_llama(q, cos_full, sin_full, trans_mat, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_llama(k, cos_full, sin_full, trans_mat, is_decode_mode=False)

        if _USE_MANUAL_ATTENTION:
            # Debug path: manual matmul/softmax/matmul instead of ttnn's SDPA op, used
            # earlier to isolate whether SDPA itself was the source of the 40-layer PCC
            # drift (it wasn't -- see functional_predictor.py's mask-handling finding,
            # which is where a manual-attention path actually turned out to matter).
            k_t = ttnn.permute(k, (0, 1, 3, 2))
            attn = ttnn.matmul(q, k_t, compute_kernel_config=_hifi_compute_kernel_config()) * self.cfg.head_dim**-0.5
            attn = ttnn.softmax(attn, dim=-1, compute_kernel_config=_hifi_compute_kernel_config())
            out = ttnn.matmul(attn, v, compute_kernel_config=_hifi_compute_kernel_config())
        else:
            # q/k/v are already bf16 (qkv weight is bf16), which is what SDPA requires anyway.
            out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, compute_kernel_config=_hifi_compute_kernel_config()
            )
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # (B, 1, N, H*D)
        out = ttnn.reshape(out, (batch, seq_len, H * D))
        out = ttnn.linear(out, self.proj_w, bias=self.proj_b, compute_kernel_config=_hifi_compute_kernel_config())
        return ttnn.typecast(out, residual_dtype)


class EncoderBlock(LightweightModule):
    """norm1 -> RoPEAttention -> residual -> norm2 -> MLP(gelu) -> residual, exactly
    matching reference `Block.forward` (encoder uses use_rope=True, plain GELU MLP --
    encoder state dict has only mlp.fc1/fc2, confirming no SwiGLU here)."""

    def __init__(self, attn: RoPEAttention, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, cfg):
        self.attn = attn
        self.norm1_w, self.norm1_b = norm1
        self.norm2_w, self.norm2_b = norm2
        self.fc1_w, self.fc1_b = fc1_w, fc1_b
        self.fc2_w, self.fc2_b = fc2_w, fc2_b
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, layer_idx: int, cfg: VJEPA2EncoderConfig, device, dtype=ttnn.float32):
        # Mixed precision: linear weights (qkv/proj/fc1/fc2) are bf16 -- that's where the
        # FLOPs and memory bandwidth are, and it's the real perf lever. norm weights and
        # the residual stream itself stay at `dtype` (fp32 by default): layer_norm and the
        # two residual adds per block are cheap elementwise ops, and this is exactly the
        # state (the accumulator that compounds over 40 layers) that a 40-block bisection
        # showed was precision-sensitive. Empirically this beats even all-fp32 (PCC 0.9968
        # vs 0.9966 at real resolution) while keeping the bf16 weight/compute win.
        prefix = f"module.blocks.{layer_idx}"
        attn = RoPEAttention.from_state_dict(state_dict, prefix=prefix, cfg=cfg, device=device, dtype=dtype)
        norm1 = _torch_norm_to_ttnn(
            state_dict[f"{prefix}.norm1.weight"], state_dict[f"{prefix}.norm1.bias"], device, dtype
        )
        norm2 = _torch_norm_to_ttnn(
            state_dict[f"{prefix}.norm2.weight"], state_dict[f"{prefix}.norm2.bias"], device, dtype
        )
        fc1_w, fc1_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.mlp.fc1.weight"], state_dict[f"{prefix}.mlp.fc1.bias"], device, ttnn.bfloat16
        )
        fc2_w, fc2_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.mlp.fc2.weight"], state_dict[f"{prefix}.mlp.fc2.bias"], device, ttnn.bfloat16
        )
        return cls(attn, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, cfg)

    def forward(self, x: "ttnn.Tensor", rope_tables: tuple, batch: int, seq_len: int) -> "ttnn.Tensor":
        eps = self.cfg.layer_norm_eps
        residual_dtype = x.dtype
        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm1_w, bias=self.norm1_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = self.attn(h, rope_tables, batch, seq_len)  # returns residual_dtype already
        x = residual + h

        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm2_w, bias=self.norm2_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = ttnn.typecast(h, ttnn.bfloat16)
        h = ttnn.linear(h, self.fc1_w, bias=self.fc1_b, compute_kernel_config=_hifi_compute_kernel_config())
        h = ttnn.gelu(h)
        h = ttnn.linear(h, self.fc2_w, bias=self.fc2_b, compute_kernel_config=_hifi_compute_kernel_config())
        h = ttnn.typecast(h, residual_dtype)
        x = residual + h
        return x


class VJEPA2Encoder(LightweightModule):
    """Full encoder: PatchEmbed3D -> 40x EncoderBlock -> final LayerNorm. Single 1x1
    device mesh; every token attends to every token in one forward pass (no cache, no
    prefill/decode split -- see module docstring for why that's the right shape here)."""

    def __init__(
        self, patch_embed: PatchEmbed3D, blocks: list, final_norm, cfg: VJEPA2EncoderConfig, device, dtype=ttnn.float32
    ):
        self.patch_embed = patch_embed
        self.blocks = blocks
        self.final_norm_w, self.final_norm_b = final_norm
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: VJEPA2EncoderConfig, device, dtype=ttnn.float32):
        patch_embed = PatchEmbed3D.from_state_dict(state_dict, cfg=cfg, device=device, dtype=dtype)
        blocks = [
            EncoderBlock.from_state_dict(state_dict, layer_idx=i, cfg=cfg, device=device, dtype=dtype)
            for i in range(cfg.num_layers)
        ]
        final_norm = _torch_norm_to_ttnn(
            state_dict["module.norm.weight"], state_dict["module.norm.bias"], device, dtype
        )
        return cls(patch_embed, blocks, final_norm, cfg, device, dtype=dtype)

    def prepare_input(self, pixel_values: torch.Tensor) -> tuple:
        """One-time host reshape + device transfer. Returns (patches_tt, grid) to feed
        into `forward_device`. Call once per (shape, buffer) -- not per iteration when
        benchmarking under trace replay."""
        patches, grid = self.patch_embed.unfold_patches(pixel_values)
        return self.patch_embed.to_device(patches), grid

    def get_rope_tables(self, gT: int, gH: int, gW: int) -> tuple:
        """Built once per (gT,gH,gW) and cached -- rope tables depend only on the grid
        shape, never on input content, so rebuilding them inside a traced/replayed loop
        would be a pointless repeated device allocation. Returns (cos_full, sin_full,
        trans_mat) for the fused `rotary_embedding_llama` path -- see
        `build_fused_rope_table`'s docstring for why one fused table reproduces the old
        per-axis d/h/w tables."""
        key = (gT, gH, gW)
        if not hasattr(self, "_rope_cache"):
            self._rope_cache = {}
        if key not in self._rope_cache:
            # bf16, matching q/k inside RoPEAttention (derived from the bf16 qkv weight),
            # not `self.dtype` (the fp32 residual-stream dtype) -- rope is applied before
            # the attention output is cast back up.
            pos_d, pos_h, pos_w = axis_positions_3d(gT, gH, gW, self.cfg)
            cos_full, sin_full = build_fused_rope_table(pos_d, pos_h, pos_w, self.cfg, self.device, dtype=ttnn.bfloat16)
            trans_mat = get_rope_trans_mat(self.device, dtype=ttnn.bfloat16)
            self._rope_cache[key] = (cos_full, sin_full, trans_mat)
        return self._rope_cache[key]

    def forward_device(self, patches_tt: "ttnn.Tensor", grid: tuple, batch: int) -> "ttnn.Tensor":
        """Pure device op graph -- patch projection + all blocks + final norm. Safe to
        capture in a trace: no torch tensors, no host reshapes, no fresh allocations for
        rope tables (reuses the cache from `get_rope_tables`)."""
        cfg = self.cfg
        gT, gH, gW = grid
        seq_len = gT * gH * gW
        tokens = self.patch_embed.forward_device(patches_tt)
        x = ttnn.reshape(tokens, (batch, seq_len, cfg.hidden_size))
        rope_tables = self.get_rope_tables(gT, gH, gW)
        for block in self.blocks:
            x = block(x, rope_tables, batch, seq_len)
        x = ttnn.layer_norm(
            x,
            weight=self.final_norm_w,
            bias=self.final_norm_b,
            epsilon=cfg.layer_norm_eps,
            compute_kernel_config=_hifi_compute_kernel_config(),
        )
        return x

    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """Convenience path for correctness tests: prepares input and runs the device
        graph every call. Not what a traced benchmark should use."""
        B = pixel_values.shape[0]
        patches_tt, grid = self.prepare_input(pixel_values)
        return self.forward_device(patches_tt, grid, B)
