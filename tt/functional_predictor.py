# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""TTNN functional bring-up of the vjepa2-ac-vitg action-conditioned predictor.

Reference: reference/src/models/ac_predictor.py (VisionTransformerPredictorAC) +
reference/src/models/utils/modules.py (ACBlock, ACRoPEAttention). Config values below
are pinned to the AC checkpoint's own training config
(reference/configs/train/vitg16/droid-256px-8f.yaml): use_extrinsics=false (the
extrinsics_encoder weight exists in the checkpoint but is never called -- matches the
reference exactly), pred_is_frame_causal=true, pred_num_heads=16, pred_depth=24.

Unlike the encoder, this is NOT full bidirectional attention: frame-causal means each
frame's tokens attend to every token in the current and all PAST frames, never future
ones -- implemented as a fixed additive attention bias, not autoregressive decode (the
whole clip's frames are still processed in one forward pass, not one at a time).

Two token kinds share one attention sequence per frame: [action_token, state_token,
<H*W spatial tokens>], repeated per frame, flattened to one sequence of length
T*(2+H*W). Action/state tokens get a temporal-only RoPE rotation (position = frame
index, only the first `rope_axis_dim` head-dim slots rotated); spatial tokens get the
same full 3-axis (depth/height/width) RoPE as the encoder, over the same T,H,W grid.

Both token kinds' rotations are applied in a SINGLE pass over the whole merged sequence
via `build_unified_rope_tables` (see its docstring) -- q/k/v are never split into
separate cond/frame tensors that would then need re-merging. An earlier version did
split them, and profiling found the re-merge step (working around a TILE_LAYOUT padding
bug on the tiny cond_tokens dimension) cost ~36% of total device time by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tt.functional_encoder import (
    _hifi_compute_kernel_config,
    _torch_linear_to_ttnn,
    _torch_norm_to_ttnn,
    build_fused_rope_table,
    get_rope_trans_mat,
)

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class VJEPA2PredictorConfig:
    encoder_hidden_size: int = 1408  # input/output dim (matches the encoder)
    pred_hidden_size: int = 1024
    pred_num_heads: int = 16
    pred_num_layers: int = 24
    pred_mlp_ratio: float = 4.0
    action_embed_dim: int = 7
    layer_norm_eps: float = 1e-6
    grid_size: int = 16  # same meaning as VJEPA2EncoderConfig.grid_size -- see its docstring
    cond_tokens: int = 2  # action + state (use_extrinsics=False for this checkpoint)

    @property
    def head_dim(self) -> int:
        return self.pred_hidden_size // self.pred_num_heads

    @property
    def rope_axis_dim(self) -> int:
        return 2 * ((self.head_dim // 3) // 2)


def build_unified_rope_tables(
    T: int, gH: int, gW: int, cond_tokens: int, cfg: VJEPA2PredictorConfig, device, dtype=ttnn.bfloat16
) -> tuple:
    """(cos_full, sin_full) spanning the full head_dim, covering the FULL merged sequence
    (cond + spatial tokens, all T frames) in one pass -- avoids ever splitting q/k/v into
    separate cond/frame tensors that then need re-merging. Built via the encoder's
    `build_fused_rope_table` (see its docstring for the per-axis block-concat + fused
    `rotary_embedding_llama` reasoning) with an `identity_mask` at cond positions.

    This is possible because cond and frame tokens' "d" (temporal) rotation is
    mathematically the *same* formula (position = frame index) for both -- the reference
    computes it via two different code paths (a per-action-slot loop vs `separate_positions`
    on the full index range) but they agree bit-for-bit. Cond tokens never rotate on the
    h/w axes at all (reference passes `q[..., d_dim:]` straight through); representing
    that as an IDENTITY rotation (cos=1, sin=0) at cond positions in the h/w tables makes
    the same fused rotation correct for every position uniformly: x*1 + rotate_half(x)*0
    == x, so cond's h/w segments end up bit-for-bit unrotated, matching the reference's
    plain pass-through exactly.

    A prior implementation split cond/frame into separate (B,H,tokens,Dh) tensors, roped
    them separately, then re-merged per frame -- profiling showed tilize/untilize alone
    (from TILE_LAYOUT padding the tiny cond_tokens=2 dim, whether via an explicit
    ROW_MAJOR round-trip or an implicit one inside SliceDeviceOperation) was ~36% of
    total device time. This version never creates that shape at all.
    """
    HW = gH * gW
    N_T = cond_tokens + HW
    N = T * N_T

    idx = torch.arange(N)
    frame_of = idx // N_T
    within = idx % N_T
    is_cond = within < cond_tokens
    frame_local = (within - cond_tokens).clamp(min=0)
    h_idx = frame_local // gW
    w_idx = frame_local % gW

    pos_d = frame_of.float()
    pos_h = h_idx.float() * (cfg.grid_size / gH)
    pos_w = w_idx.float() * (cfg.grid_size / gW)

    return build_fused_rope_table(pos_d, pos_h, pos_w, cfg, device, dtype=dtype, identity_mask=is_cond)


def build_frame_causal_mask(T: int, HW: int, cond_tokens: int, dtype=torch.float32) -> torch.Tensor:
    """Additive attention bias: frame t attends to every token (cond + spatial) in
    frames 0..t, nothing in t+1..T-1. Matches reference `build_action_block_causal_attention_mask`
    (boolean there; converted to additive here since ttnn's SDPA attn_mask is additive,
    same convention as models/tt_transformers/tt/common.py's causal masks)."""
    N_T = cond_tokens + HW
    N = T * N_T
    allowed = torch.zeros(N, N, dtype=torch.bool)
    block = torch.ones(N_T, N_T, dtype=torch.bool)
    for t1 in range(T):
        for t2 in range(t1 + 1):
            allowed[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = block
    bias = torch.zeros(N, N, dtype=dtype)
    bias.masked_fill_(~allowed, -30000.0)
    return bias.reshape(1, 1, N, N)


class ACRoPEAttention(LightweightModule):
    def __init__(self, qkv_w, qkv_b, proj_w, proj_b, cfg: VJEPA2PredictorConfig, device):
        self.qkv_w, self.qkv_b = qkv_w, qkv_b
        self.proj_w, self.proj_b = proj_w, proj_b
        self.cfg = cfg
        self.device = device

    @classmethod
    def from_state_dict(cls, state_dict, *, prefix: str, cfg: VJEPA2PredictorConfig, device, dtype=ttnn.float32):
        # Weights always bf16 -- see the mixed-precision note on PredictorBlock. `dtype`
        # only controls the activation/residual side.
        qkv_w, qkv_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.attn.qkv.weight"], state_dict[f"{prefix}.attn.qkv.bias"], device, ttnn.bfloat16
        )
        proj_w, proj_b = _torch_linear_to_ttnn(
            state_dict[f"{prefix}.attn.proj.weight"], state_dict[f"{prefix}.attn.proj.bias"], device, ttnn.bfloat16
        )
        return cls(qkv_w, qkv_b, proj_w, proj_b, cfg, device)

    def forward(
        self, x: "ttnn.Tensor", rope_tables: tuple, attn_mask: "ttnn.Tensor", B: int, T: int, HW: int
    ) -> "ttnn.Tensor":
        """x arrives at the block's residual dtype (fp32 by default); cast down to bf16
        for qkv/attention/proj compute (weights are bf16), back up before returning."""
        cfg = self.cfg
        H, Dh = cfg.pred_num_heads, cfg.head_dim
        cond_tokens = cfg.cond_tokens
        N_T = cond_tokens + HW
        N = T * N_T
        residual_dtype = x.dtype
        x = ttnn.typecast(x, ttnn.bfloat16)

        # No cond/frame split: `rope_tables` is the unified per-position (cos_full,
        # sin_full, trans_mat) built by `build_unified_rope_tables` covering the whole
        # merged sequence in one pass -- identical in spirit to the encoder's flat
        # RoPEAttention. See that function's docstring for why cond and frame tokens can
        # share one fused rotation pass (same "d" formula for both; h/w are
        # identity-rotated at cond positions, which is bit-for-bit equivalent to the
        # reference's plain pass-through there).
        # Fused split-into-heads (see the matching note in functional_encoder.py's
        # RoPEAttention): profiling found ~95% of device time in reshape/tilize/slice/
        # permute, ~2% in matmul. `nlp_create_qkv_heads` replaces the manual
        # reshape+slice+permute chain with tt-metal's own primitive for it.
        qkv = ttnn.linear(x, self.qkv_w, bias=self.qkv_b, compute_kernel_config=_hifi_compute_kernel_config())
        qkv = ttnn.reshape(qkv, (B, 1, N, 3 * H * Dh))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, transpose_k_heads=False)

        # Fused rotary embedding (see functional_encoder.py's RoPEAttention for the same
        # pattern and its verification): one device call per q/k spanning the whole
        # head_dim instead of a per-axis slice -> rotate -> concat loop.
        cos_full, sin_full, trans_mat = rope_tables
        q = ttnn.experimental.rotary_embedding_llama(q, cos_full, sin_full, trans_mat, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_llama(k, cos_full, sin_full, trans_mat, is_decode_mode=False)

        # ttnn.transformer.scaled_dot_product_attention's attn_mask feature does not
        # produce correct results for this frame-causal mask (a large, block-structured
        # NxN bias, not a simple diagonal-causal or sliding-window pattern): isolated
        # against a verified-correct q/k/v, masked SDPA gave PCC 0.927 vs an unmasked-SDPA
        # baseline of 0.997 -- and the masked PCC didn't move at all across mask
        # magnitudes (-30000, -1e6, -inf all identical), which rules out a value/overflow
        # issue and points to the kernel not consuming the mask correctly at this
        # shape/granularity. Manual matmul+softmax+matmul with the same additive mask
        # gives PCC 0.9999 against the same reference -- use that instead. (SDPA itself
        # is fine and preferred where no mask is needed, e.g. the encoder.)
        scale = Dh**-0.5
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        scores = ttnn.matmul(q, k_t, compute_kernel_config=_hifi_compute_kernel_config()) * scale
        scores = scores + attn_mask
        attn_w = ttnn.softmax(scores, dim=-1, compute_kernel_config=_hifi_compute_kernel_config())
        out = ttnn.matmul(attn_w, v, compute_kernel_config=_hifi_compute_kernel_config())
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # (B, 1, N, H*Dh)
        out = ttnn.reshape(out, (B, N, H * Dh))
        out = ttnn.linear(out, self.proj_w, bias=self.proj_b, compute_kernel_config=_hifi_compute_kernel_config())
        return ttnn.typecast(out, residual_dtype)


class PredictorBlock(LightweightModule):
    """norm1 -> ACRoPEAttention -> residual -> norm2 -> MLP(gelu) -> residual. Same
    shape as the encoder's EncoderBlock; only the attention module differs."""

    def __init__(self, attn: ACRoPEAttention, norm1, norm2, fc1_w, fc1_b, fc2_w, fc2_b, cfg):
        self.attn = attn
        self.norm1_w, self.norm1_b = norm1
        self.norm2_w, self.norm2_b = norm2
        self.fc1_w, self.fc1_b = fc1_w, fc1_b
        self.fc2_w, self.fc2_b = fc2_w, fc2_b
        self.cfg = cfg

    @classmethod
    def from_state_dict(cls, state_dict, *, layer_idx: int, cfg: VJEPA2PredictorConfig, device, dtype=ttnn.float32):
        # Same mixed-precision split as EncoderBlock: bf16 linear weights (qkv/proj/fc1/
        # fc2 -- where the FLOPs/bandwidth are), fp32 norm weights and residual stream
        # (`dtype`, where the 24-layer-depth precision sensitivity actually lives).
        prefix = f"module.predictor_blocks.{layer_idx}"
        attn = ACRoPEAttention.from_state_dict(state_dict, prefix=prefix, cfg=cfg, device=device, dtype=dtype)
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

    def forward(self, x, rope_tables, attn_mask, B, T, HW):
        eps = self.cfg.layer_norm_eps
        residual_dtype = x.dtype
        residual = x
        h = ttnn.layer_norm(
            x, weight=self.norm1_w, bias=self.norm1_b, epsilon=eps, compute_kernel_config=_hifi_compute_kernel_config()
        )
        h = self.attn(h, rope_tables, attn_mask, B, T, HW)  # returns residual_dtype already
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


class VJEPA2Predictor(LightweightModule):
    """predictor_embed -> interleave [action, state] tokens per frame -> 24x
    PredictorBlock (frame-causal) -> drop cond tokens -> predictor_norm -> predictor_proj
    back to encoder_hidden_size."""

    def __init__(
        self,
        embed_w,
        embed_b,
        action_w,
        action_b,
        state_w,
        state_b,
        blocks: list,
        final_norm,
        proj_w,
        proj_b,
        cfg: VJEPA2PredictorConfig,
        device,
        dtype=ttnn.float32,
    ):
        self.embed_w, self.embed_b = embed_w, embed_b
        self.action_w, self.action_b = action_w, action_b
        self.state_w, self.state_b = state_w, state_b
        self.blocks = blocks
        self.final_norm_w, self.final_norm_b = final_norm
        self.proj_w, self.proj_b = proj_w, proj_b
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_state_dict(cls, state_dict, *, cfg: VJEPA2PredictorConfig, device, dtype=ttnn.float32):
        # bf16 weights (mixed-precision default); ttnn.linear tolerates a higher-precision
        # (fp32) activation against a bf16 weight without complaint (verified on the
        # encoder's patch-embed projection), so the input/output activation stays at
        # `dtype` on both sides of these two boundary projections.
        embed_w, embed_b = _torch_linear_to_ttnn(
            state_dict["module.predictor_embed.weight"],
            state_dict["module.predictor_embed.bias"],
            device,
            ttnn.bfloat16,
        )
        action_w, action_b = _torch_linear_to_ttnn(
            state_dict["module.action_encoder.weight"], state_dict["module.action_encoder.bias"], device, ttnn.bfloat16
        )
        state_w, state_b = _torch_linear_to_ttnn(
            state_dict["module.state_encoder.weight"], state_dict["module.state_encoder.bias"], device, ttnn.bfloat16
        )
        # module.extrinsics_encoder exists in the checkpoint but use_extrinsics=False for
        # this training config -- never called, matching the reference exactly.
        blocks = [
            PredictorBlock.from_state_dict(state_dict, layer_idx=i, cfg=cfg, device=device, dtype=dtype)
            for i in range(cfg.pred_num_layers)
        ]
        final_norm = _torch_norm_to_ttnn(
            state_dict["module.predictor_norm.weight"], state_dict["module.predictor_norm.bias"], device, dtype
        )
        proj_w, proj_b = _torch_linear_to_ttnn(
            state_dict["module.predictor_proj.weight"], state_dict["module.predictor_proj.bias"], device, ttnn.bfloat16
        )
        return cls(
            embed_w,
            embed_b,
            action_w,
            action_b,
            state_w,
            state_b,
            blocks,
            final_norm,
            proj_w,
            proj_b,
            cfg,
            device,
            dtype=dtype,
        )

    def prepare_conditioning(self, actions: torch.Tensor, states: torch.Tensor) -> tuple:
        """One-time host->device transfer for the action/state conditioning tensors.
        Call once per buffer, not per traced-replay iteration."""
        a_tt = ttnn.from_torch(actions.unsqueeze(2), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        s_tt = ttnn.from_torch(states.unsqueeze(2), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        return a_tt, s_tt

    def get_rope_and_mask(self, gT: int, gH: int, gW: int) -> tuple:
        """Built once per (gT,gH,gW) and cached, same reasoning as the encoder's
        `get_rope_tables`: these depend only on grid shape, not on input content, so
        rebuilding them inside a traced/replayed loop is a pointless repeated allocation.

        Bounded to a handful of entries: `attn_mask` is O((gT*HW)^2) and a caller doing
        an ever-growing-context rollout (e.g. the imagination-rollout demo, which visits
        a new, never-repeated gT every step) would otherwise accumulate one such mask
        PER STEP, permanently, for the life of this object -- found the hard way via a
        DRAM OOM roughly 60-70 distinct shapes into a long rollout. A fixed-shape caller
        (benchmark.py's traced replay) only ever uses one key, so this cap changes
        nothing for it."""
        key = (gT, gH, gW)
        if not hasattr(self, "_rope_mask_cache"):
            self._rope_mask_cache = {}
        _MAX_CACHE_ENTRIES = 2
        if key not in self._rope_mask_cache and len(self._rope_mask_cache) >= _MAX_CACHE_ENTRIES:
            oldest_key = next(iter(self._rope_mask_cache))
            del self._rope_mask_cache[oldest_key]
        if key not in self._rope_mask_cache:
            # bf16, matching q/k/scores inside ACRoPEAttention (derived from bf16 qkv
            # weight), not `self.dtype` (the fp32 residual-stream dtype).
            HW = gH * gW
            cond_tokens = self.cfg.cond_tokens
            cos_full, sin_full = build_unified_rope_tables(
                gT, gH, gW, cond_tokens, self.cfg, self.device, dtype=ttnn.bfloat16
            )
            trans_mat = get_rope_trans_mat(self.device, dtype=ttnn.bfloat16)
            rope_tables = (cos_full, sin_full, trans_mat)
            mask_bias = build_frame_causal_mask(gT, HW, cond_tokens)
            attn_mask = ttnn.from_torch(mask_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            self._rope_mask_cache[key] = (rope_tables, attn_mask)
        return self._rope_mask_cache[key]

    def forward_device(
        self,
        context_tokens: "ttnn.Tensor",
        a_tt: "ttnn.Tensor",
        s_tt: "ttnn.Tensor",
        gT: int,
        gH: int,
        gW: int,
        batch: int,
    ) -> "ttnn.Tensor":
        """Pure device op graph -- embed/interleave + all blocks + norm + proj. Safe to
        capture in a trace: no torch tensors, no host reshapes, no fresh rope/mask
        allocations (reuses the cache from `get_rope_and_mask`)."""
        cfg = self.cfg
        B = batch
        HW = gH * gW
        cond_tokens = cfg.cond_tokens

        x = ttnn.linear(
            context_tokens, self.embed_w, bias=self.embed_b, compute_kernel_config=_hifi_compute_kernel_config()
        )  # (B, T*HW, pred_hidden)
        x = ttnn.reshape(x, (B, gT, HW, cfg.pred_hidden_size))

        a = ttnn.linear(a_tt, self.action_w, bias=self.action_b, compute_kernel_config=_hifi_compute_kernel_config())
        s = ttnn.linear(s_tt, self.state_w, bias=self.state_b, compute_kernel_config=_hifi_compute_kernel_config())
        # a, s: (B, T, 1, pred_hidden) -- concat with x along the token axis: [action, state, frame...]
        x = ttnn.concat([a, s, x], dim=2)  # (B, T, cond_tokens+HW, pred_hidden)
        x = ttnn.reshape(x, (B, gT * (cond_tokens + HW), cfg.pred_hidden_size))

        rope_tables, attn_mask = self.get_rope_and_mask(gT, gH, gW)

        for block in self.blocks:
            x = block(x, rope_tables, attn_mask, B, gT, HW)

        x = ttnn.reshape(x, (B, gT, cond_tokens + HW, cfg.pred_hidden_size))
        x = x[:, :, cond_tokens:]  # drop action/state tokens, keep frame predictions
        x = ttnn.reshape(x, (B, gT * HW, cfg.pred_hidden_size))

        x = ttnn.layer_norm(
            x,
            weight=self.final_norm_w,
            bias=self.final_norm_b,
            epsilon=cfg.layer_norm_eps,
            compute_kernel_config=_hifi_compute_kernel_config(),
        )
        x = ttnn.linear(x, self.proj_w, bias=self.proj_b, compute_kernel_config=_hifi_compute_kernel_config())
        return x

    def forward(
        self, context_tokens: "ttnn.Tensor", actions: torch.Tensor, states: torch.Tensor, gT: int, gH: int, gW: int
    ) -> "ttnn.Tensor":
        """Convenience path for correctness tests: prepares conditioning tensors and runs
        the device graph every call. Not what a traced benchmark should use -- see
        `prepare_conditioning`/`forward_device`."""
        B = actions.shape[0]
        a_tt, s_tt = self.prepare_conditioning(actions, states)
        return self.forward_device(context_tokens, a_tt, s_tt, gT, gH, gW, B)
