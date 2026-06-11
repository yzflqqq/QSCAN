"""Semantic-conditioning modules for QSCAN.

These layers infer lightweight per-patch semantic priors (content tags, a
texture/structure routing weight, and a spatial importance map) and use them to
modulate the restoration features. All modulation layers are zero-initialized so
the network starts from an unmodulated baseline and learns to inject guidance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qscan.archs.layers import LayerNorm


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maxv, _ = x.max(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))


class SpatialChannelFusion(nn.Module):
    """Dual channel/spatial attention fusion module."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.proj_c = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_s = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        chan = self.proj_c(self.ca(x_norm) * x_norm)
        spat = self.proj_s(self.sa(x_norm) * x_norm)
        return chan + spat


class PerceptualFieldGate(nn.Module):
    """Multi-scale depthwise branches gated by local gradient statistics."""

    def __init__(self, dim: int):
        super().__init__()
        self.branch_local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.branch_mid = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.branch_periph = nn.Conv2d(dim, dim, 7, 1, 6, dilation=2, groups=dim, bias=False)
        self.branch_center = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1, 1, 0),
        )
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.fuse = nn.Conv2d(dim, dim, 1, 1, 0)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)
        self.register_buffer('laplace', laplace, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(src, self.sobel_x, padding=1).abs()
        gy = F.conv2d(src, self.sobel_y, padding=1).abs()
        lap = F.conv2d(src, self.laplace, padding=1).abs()
        mean = F.avg_pool2d(src, 3, 1, 1)
        var = torch.clamp(F.avg_pool2d(src * src, 3, 1, 1) - mean * mean, min=0.0)
        gate = torch.softmax(self.gate(torch.cat([gx, gy, lap, var], dim=1)), dim=1)

        local = self.branch_local(x)
        mid = self.branch_mid(x)
        periph = self.branch_periph(x) - torch.tanh(self.beta) * self.branch_center(x)
        mix = gate[:, 0:1] * local + gate[:, 1:2] * mid + gate[:, 2:3] * periph
        return self.fuse(mix)


class TextureExpert(nn.Module):
    """Texture-restoration expert combining the field gate and channel/spatial fusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.pfg = PerceptualFieldGate(dim)
        self.scfm = SpatialChannelFusion(dim)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pfg(x)
        y = y + self.scfm(y)
        y = self.out(y)
        return torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)


class SpatialFrequencyExpert(nn.Module):
    """Structure-restoration expert mixing a spatial path and a frequency path."""

    def __init__(self, dim: int, freq_clamp: float = 32.0):
        super().__init__()
        self.freq_clamp = freq_clamp
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )
        self.freq_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
        )
        self.fuse = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            ffted = torch.fft.rfft2(x.float(), norm='ortho')
            freq = torch.cat([ffted.real, ffted.imag], dim=1)
            freq = torch.clamp(freq, min=-self.freq_clamp, max=self.freq_clamp)
            freq = self.freq_proj(freq)
            real, imag = torch.chunk(freq, 2, dim=1)
            ffted = torch.complex(real, imag)
            freq = torch.fft.irfft2(ffted, s=x.shape[-2:], norm='ortho')
            freq = torch.nan_to_num(freq, nan=0.0, posinf=1e3, neginf=-1e3)
        freq = freq.to(x.dtype)
        out = self.fuse(torch.cat([spatial, freq], dim=1))
        return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)


class SemanticRouter(nn.Module):
    """Predicts content tags, an expert routing weight, and a spatial importance map."""

    def __init__(self, dim: int, num_tags: int, hidden_dim: int = 64, logit_clamp: float = 20.0):
        super().__init__()
        self.logit_clamp = logit_clamp
        self.norm = LayerNorm(dim)
        self.importance_head = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1, 1, 0),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.tag_head = nn.Linear(hidden_dim, num_tags)
        self.route_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        avg = F.adaptive_avg_pool2d(x_norm, 1).flatten(1)
        maxv = F.adaptive_max_pool2d(x_norm, 1).flatten(1)
        pooled = self.pool_proj(torch.cat([avg, maxv], dim=1))
        tag_logits = self.tag_head(pooled)
        route_logits = self.route_head(pooled)
        importance_logits = self.importance_head(x_norm)
        c = self.logit_clamp
        tag_logits = torch.clamp(torch.nan_to_num(tag_logits, nan=0.0, posinf=c, neginf=-c), min=-c, max=c)
        route_logits = torch.clamp(torch.nan_to_num(route_logits, nan=0.0, posinf=c, neginf=-c), min=-c, max=c)
        importance_logits = torch.clamp(
            torch.nan_to_num(importance_logits, nan=0.0, posinf=c, neginf=-c), min=-c, max=c
        )
        return tag_logits, route_logits, importance_logits


class SemanticExpertFusion(nn.Module):
    """Fuses texture/structure experts under semantic routing into a residual delta."""

    def __init__(
        self,
        dim: int,
        num_tags: int,
        delta_clip: float = 0.25,
        tag_gain_limit: float = 1.0,
        freq_clamp: float = 32.0,
        logit_clamp: float = 20.0,
    ):
        super().__init__()
        self.router = SemanticRouter(dim, num_tags, hidden_dim=max(dim, 64), logit_clamp=logit_clamp)
        self.texture_expert = TextureExpert(dim)
        self.structure_expert = SpatialFrequencyExpert(dim, freq_clamp=freq_clamp)
        self.tag_proj = nn.Linear(num_tags, dim)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.delta_clip = delta_clip
        self.tag_gain_limit = tag_gain_limit
        nn.init.zeros_(self.tag_proj.weight)
        nn.init.zeros_(self.tag_proj.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            tag_logits, route_logits, importance_logits = self.router(x_fp32)

            route_prob = torch.softmax(route_logits, dim=1)
            texture = self.texture_expert(x_fp32)
            structure = self.structure_expert(x_fp32)

            mix = (
                route_prob[:, 0:1].view(x.shape[0], 1, 1, 1) * texture +
                route_prob[:, 1:2].view(x.shape[0], 1, 1, 1) * structure
            )
            tag_gain = self.tag_proj(torch.sigmoid(tag_logits)).view(x.shape[0], x.shape[1], 1, 1)
            tag_gain = self.tag_gain_limit * torch.tanh(tag_gain)
            mix = mix * (1.0 + tag_gain)
            gated_mix = torch.sigmoid(importance_logits) * mix
            delta = self.delta_clip * torch.tanh(self.out(gated_mix))
            delta = torch.nan_to_num(delta, nan=0.0, posinf=self.delta_clip, neginf=-self.delta_clip)

        delta = delta.to(x.dtype)
        aux = {
            'tag_logits': tag_logits,
            'route_logits': route_logits,
            'importance_logits': importance_logits,
        }
        return delta, aux
