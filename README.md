# QSCAN

**Quality-aware Semantic Conv-Attention Network** for efficient image super-resolution.

QSCAN is a lightweight conv-attention super-resolution backbone augmented with
two semantic-conditioning paths:

1. **Residual expert fusion** — at selected depths, a semantic router predicts
   per-patch content tags, a texture/structure routing weight, and a spatial
   importance map. These drive a texture expert and a spatial-frequency
   structure expert whose fused output is injected as a small residual delta.
2. **Semantic-conditioned conv-attention** — the deepest blocks modulate their
   convolutional-attention path with the same semantic priors (per-channel gain
   and bias, plus a spatial importance gate).

Every semantic-modulation layer is zero-initialized, so an untrained network
reduces exactly to a plain conv-attention super-resolution backbone and learns
to inject guidance from there.

This repository contains **only the model definition** — no training,
evaluation, or data pipelines.

## Install

```bash
pip install -e .
```

Requires PyTorch ≥ 2.5 (the model uses `torch.nn.attention.flex_attention`) and
`einops`.

## Usage

```python
import torch
from qscan import QSCAN

model = QSCAN(
    dim=64,
    pdim=16,
    kernel_size=13,
    n_blocks=5,
    conv_blocks=5,
    window_size=32,
    num_heads=4,
    upscaling_factor=2,
    exp_ratio=1.25,
    attn_type='Naive',   # 'Naive' | 'SDPA' | 'Flex' | 'FlashBias'
)

lr = torch.randn(1, 3, 64, 64)
sr = model(lr)                       # (1, 3, 128, 128)
sr, aux = model(lr, return_aux=True) # aux holds the averaged semantic logits
```

### Attention backends

| Backend     | Notes                                                        |
|-------------|--------------------------------------------------------------|
| `Naive`     | Numerically stable reference; works everywhere.              |
| `Flex`      | Fast and memory efficient (recommended on Linux).            |
| `SDPA`      | Memory-efficient kernel; good fallback on Windows.           |
| `FlashBias` | Flash Attention with low-rank relative-position bias; infer. |

### Auxiliary semantic outputs

Calling `model(x, return_aux=True)` returns `(output, aux)` where `aux` is a
dict of depth-averaged `tag_logits`, `route_logits`, and `importance_logits`.
These can be supervised with auxiliary losses if you have semantic labels, or
simply ignored.

## Model registry

`QSCAN` registers itself in a minimal built-in `ARCH_REGISTRY`
(`qscan.archs.registry`). If you integrate with a framework that has its own
registry, just import `QSCAN` directly and ignore it.

## License

MIT. See [LICENSE](LICENSE).
