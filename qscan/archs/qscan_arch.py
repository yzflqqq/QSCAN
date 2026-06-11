"""QSCAN: Quality-aware Semantic Conv-Attention Network.

A single-file model definition that combines a windowed conv-attention backbone
with lightweight semantic conditioning. The network predicts per-patch semantic
priors (content tags, a texture/structure routing weight, and a spatial
importance map) and uses them both to (a) inject a residual expert-fusion delta
at selected depths and (b) modulate the conv-attention path in the deepest
blocks. All semantic-modulation layers are zero-initialized so an untrained
network reduces to a plain conv-attention super-resolution backbone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention

from qscan.archs.registry import ARCH_REGISTRY
from qscan.archs.layers import (
    ATTN_TYPE,
    ConvAttnWrapper,
    ConvFFN,
    ConvolutionalAttention,
    LayerNorm,
    WindowAttention,
    attention,
    geo_ensemble,
)
from qscan.archs.semantic import SemanticExpertFusion, SemanticRouter
from qscan.semantic_tags import SEMANTIC_TAGS


def _resolve_attn_func(attn_type: ATTN_TYPE):
    if attn_type == 'Naive':
        return attention
    if attn_type in ('SDPA', 'FlashBias'):
        return F.scaled_dot_product_attention
    if attn_type == 'Flex':
        return torch.compile(flex_attention, dynamic=True)
    raise NotImplementedError(f'Attention type {attn_type} is not supported.')


class Block(nn.Module):
    """Standard conv-attention block (no semantic conditioning)."""

    def __init__(
        self, dim: int, pdim: int, conv_blocks: int,
        kernel_size: int, window_size: int, num_heads: int, exp_ratio: float,
        attn_func=None, attn_type: ATTN_TYPE = 'Flex', use_ln: bool = False,
        flashbias_rank: Optional[int] = None
    ):
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 2)

        self.ln_attn = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type, flashbias_rank=flashbias_rank)

        self.lns = nn.ModuleList([LayerNorm(dim) if use_ln else nn.Identity() for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList([ConvAttnWrapper(dim, pdim, kernel_size) for _ in range(conv_blocks)])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])

        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, plk_filter: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        for ln, pconv, convffn in zip(self.lns, self.pconvs, self.convffns):
            x = x + pconv(convffn(ln(x)), plk_filter)
        x = self.conv_out(self.ln_out(x))
        return x + skip


class SemanticConvAttnWrapper(nn.Module):
    """Conv-attention path with lightweight semantic-conditioned modulation.

    The conv-attention sub-layers (plk/aggr) match the plain backbone, while the
    semantic modulation layers are zero-initialized so the module starts from the
    original image-only conv-attention behavior and learns to inject guidance.
    """

    def __init__(
        self,
        dim: int,
        pdim: int,
        kernel_size: int = 13,
        num_tags: int = len(SEMANTIC_TAGS),
        cond_hidden: int = 64,
        gain_limit: float = 0.10,
        bias_limit: float = 0.05,
    ):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, kernel_size)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gain_limit = gain_limit
        self.bias_limit = bias_limit

        cond_dim = num_tags + 2
        hidden = max(cond_hidden, dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 2),
        )
        self.importance_proj = nn.Conv2d(1, dim, 1, 1, 0)

        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)
        nn.init.zeros_(self.importance_proj.weight)
        nn.init.zeros_(self.importance_proj.bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor, prior: Optional[dict] = None) -> torch.Tensor:
        y = self.plk(x, lk_filter)
        y = self.aggr(y)
        if prior is None:
            return y

        tag_prob = torch.sigmoid(prior['tag_logits']).to(dtype=y.dtype, device=y.device)
        route_prob = torch.softmax(prior['route_logits'], dim=1).to(dtype=y.dtype, device=y.device)
        cond = torch.cat([tag_prob, route_prob], dim=1)
        gamma, beta = torch.chunk(self.cond_proj(cond), 2, dim=1)
        gamma = self.gain_limit * torch.tanh(gamma).view(y.shape[0], y.shape[1], 1, 1)
        beta = self.bias_limit * torch.tanh(beta).view(y.shape[0], y.shape[1], 1, 1)

        importance = torch.sigmoid(prior['importance_logits']).to(dtype=y.dtype, device=y.device)
        if importance.shape[-2:] != y.shape[-2:]:
            importance = F.interpolate(importance, size=y.shape[-2:], mode='bilinear', align_corners=False)
        spatial_gain = self.gain_limit * torch.tanh(self.importance_proj(importance))

        y = y * (1.0 + gamma + spatial_gain) + beta * importance
        return torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)


class SemanticBlock(nn.Module):
    """Semantic-conditioned counterpart of :class:`Block`."""

    def __init__(
        self,
        dim: int,
        pdim: int,
        conv_blocks: int,
        kernel_size: int,
        window_size: int,
        num_heads: int,
        exp_ratio: float,
        attn_func=None,
        attn_type: ATTN_TYPE = 'Flex',
        use_ln: bool = False,
        flashbias_rank: Optional[int] = None,
        num_tags: int = len(SEMANTIC_TAGS),
        sc_gain_limit: float = 0.10,
        sc_bias_limit: float = 0.05,
        logit_clamp: float = 8.0,
    ):
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 2)

        self.ln_attn = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type, flashbias_rank=flashbias_rank)

        self.router = SemanticRouter(dim, num_tags, hidden_dim=max(dim, 64), logit_clamp=logit_clamp)
        self.lns = nn.ModuleList([LayerNorm(dim) if use_ln else nn.Identity() for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList([
            SemanticConvAttnWrapper(
                dim,
                pdim,
                kernel_size,
                num_tags=num_tags,
                cond_hidden=max(dim, 64),
                gain_limit=sc_gain_limit,
                bias_limit=sc_bias_limit,
            )
            for _ in range(conv_blocks)
        ])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])

        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, plk_filter: torch.Tensor, return_aux: bool = False):
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            tag_logits, route_logits, importance_logits = self.router(x.float())
        prior = {
            'tag_logits': tag_logits,
            'route_logits': route_logits,
            'importance_logits': importance_logits,
        }

        for ln, pconv, convffn in zip(self.lns, self.pconvs, self.convffns):
            x = x + pconv(convffn(ln(x)), plk_filter, prior)
        x = self.conv_out(self.ln_out(x))
        x = torch.nan_to_num(x + skip, nan=0.0, posinf=1e3, neginf=-1e3)
        if return_aux:
            return x, prior
        return x


@ARCH_REGISTRY.register()
class QSCAN(nn.Module):
    """Quality-aware Semantic Conv-Attention Network.

    A conv-attention super-resolution backbone augmented with two semantic paths:

    * a residual :class:`SemanticExpertFusion` delta injected after the blocks
      listed in ``semantic_insert_indices`` (default: the last two blocks);
    * semantic-conditioned conv-attention in the blocks listed in
      ``semantic_block_indices`` (default: the last two blocks), which replace
      the plain :class:`Block` with :class:`SemanticBlock`.

    With every semantic-modulation layer zero-initialized, an untrained network
    behaves exactly like the plain conv-attention backbone.
    """

    def __init__(
        self,
        dim: int,
        pdim: int,
        kernel_size: int,
        n_blocks: int,
        conv_blocks: int,
        window_size: int,
        num_heads: int,
        upscaling_factor: int,
        exp_ratio: float = 2,
        attn_type: ATTN_TYPE = 'Flex',
        use_ln: bool = False,
        flashbias_rank: Optional[int] = None,
        semantic_insert_indices=None,
        num_semantic_inserts: int = 2,
        num_tags: int = len(SEMANTIC_TAGS),
        fusion_scale_limit: float = 0.25,
        fusion_scale_init: float = 0.0,
        delta_clip: float = 0.25,
        tag_gain_limit: float = 1.0,
        freq_clamp: float = 32.0,
        logit_clamp: float = 20.0,
        semantic_block_indices=None,
        num_semantic_blocks: int = 2,
        sc_gain_limit: float = 0.10,
        sc_bias_limit: float = 0.05,
    ):
        super().__init__()
        attn_func = _resolve_attn_func(attn_type)

        # Re-parameterizable large-kernel filter shared across blocks.
        self.plk_func = geo_ensemble
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        # Orthogonal init stabilizes early training of the large-kernel filter.
        torch.nn.init.orthogonal_(self.plk_filter)

        self.proj = nn.Conv2d(3, dim, 3, 1, 1)

        # Decide which blocks get semantic conditioning (default: the deepest ones).
        if semantic_block_indices is None:
            start_idx = max(0, n_blocks - min(num_semantic_blocks, n_blocks))
            semantic_block_indices = list(range(start_idx, n_blocks))
        self.semantic_block_indices = list(semantic_block_indices)

        blocks = []
        for block_idx in range(n_blocks):
            if block_idx in self.semantic_block_indices:
                blocks.append(SemanticBlock(
                    dim, pdim, conv_blocks, kernel_size, window_size, num_heads, exp_ratio,
                    attn_func=attn_func, attn_type=attn_type, use_ln=use_ln,
                    flashbias_rank=flashbias_rank, num_tags=num_tags,
                    sc_gain_limit=sc_gain_limit, sc_bias_limit=sc_bias_limit,
                    logit_clamp=logit_clamp,
                ))
            else:
                blocks.append(Block(
                    dim, pdim, conv_blocks, kernel_size, window_size, num_heads, exp_ratio,
                    attn_func, attn_type, use_ln=use_ln, flashbias_rank=flashbias_rank,
                ))
        self.blocks = nn.ModuleList(blocks)

        self.last = nn.Conv2d(dim, dim, 3, 1, 1)
        self.to_img = nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1)
        self.upscaling_factor = upscaling_factor

        # Residual expert-fusion path injected at selected depths.
        if semantic_insert_indices is None:
            start_idx = max(0, n_blocks - min(num_semantic_inserts, n_blocks))
            semantic_insert_indices = list(range(start_idx, n_blocks))
        self.semantic_insert_indices = list(semantic_insert_indices)
        self.expert_fusion = SemanticExpertFusion(
            dim, num_tags=num_tags, delta_clip=delta_clip,
            tag_gain_limit=tag_gain_limit, freq_clamp=freq_clamp, logit_clamp=logit_clamp,
        )
        self.fusion_scales = nn.Parameter(torch.empty(len(self.semantic_insert_indices), dim))
        nn.init.constant_(self.fusion_scales, fusion_scale_init)
        self.fusion_scale_limit = fusion_scale_limit

    @torch.no_grad()
    def convert(self):
        """Fold the geometric re-parameterization into the stored filter."""
        self.plk_filter = nn.Parameter(self.plk_func(self.plk_filter))
        self.plk_func = nn.Identity()

    @staticmethod
    def _merge_aux(aux_list):
        if not aux_list:
            return None
        merged = {}
        for key in aux_list[0].keys():
            merged[key] = torch.stack([aux[key] for aux in aux_list], dim=0).mean(dim=0)
        return merged

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feat = self.proj(x)
        skip = feat
        plk_filter = self.plk_func(self.plk_filter)
        insert_map = {idx: pos for pos, idx in enumerate(self.semantic_insert_indices)}
        aux_list = []

        for block_idx, block in enumerate(self.blocks):
            if isinstance(block, SemanticBlock):
                feat, aux = block(feat, plk_filter, return_aux=True)
                aux_list.append(aux)
            else:
                feat = block(feat, plk_filter)

            if block_idx in insert_map:
                delta, aux = self.expert_fusion(feat)
                scale = self.fusion_scale_limit * torch.tanh(
                    self.fusion_scales[insert_map[block_idx]]
                ).view(1, feat.shape[1], 1, 1)
                feat = torch.nan_to_num(feat + scale * delta, nan=0.0, posinf=1e3, neginf=-1e3)
                aux_list.append(aux)

        feat = self.last(feat) + skip
        out = self.to_img(feat) + torch.repeat_interleave(x, self.upscaling_factor ** 2, dim=1)
        out = F.pixel_shuffle(out, self.upscaling_factor)
        if return_aux:
            return out, self._merge_aux(aux_list)
        return out
