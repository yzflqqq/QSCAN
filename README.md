# QSCAN

**Qwen-guided Semantic Conv-Attention Network** for single-image super-resolution.

QSCAN is an efficient conv-attention SR backbone whose *late* restoration stages
are refined by compact semantic priors. During training, a vision-language model
(Qwen-9B) is run **offline** on high-resolution patches to distill three compact
priors — a content **tag** prior, a texture/structure **route** prior, and a
spatial **importance** prior. These cached priors supervise lightweight router
heads inside the network. At inference the VLM is **not needed**: the network
estimates the priors itself (train-time semantic supervision, test-time
self-routing), so QSCAN stays a single, unified SR model with no extra
inference-time dependency.

This repository contains **only the network definition** (the deployed
inference model). The offline VLM prior distillation, the semantic cache, the
two-stage training schedule, datasets, and checkpoints are **not** included.

## Method overview

```
LR ─► Shallow Embed ─► [ 3 × restoration stage ] ─► [ 2 × semantic stage ] ─► Recon Head ─► HR
                              (image-driven)            (late-stage semantic
       └────────────────── global residual ──────────────────────┘   injection)
```

* **Restoration stages (early).** Standard residual blocks: windowed
  self-attention + a convolutional FFN + a re-parameterizable large-kernel
  convolutional attention. No semantic conditioning.
* **Semantic stages (late).** Same backbone block, plus a *semantic router* that
  predicts tag / route / importance from the current features. The router drives
  two paths:
  1. **Semantic-conditioned conv-attention** — a global affine modulation
     `(γ, β)` from the tag/route priors and an importance-aware spatial gain.
  2. **Semantic expert fusion** — a *texture expert* and a *spatial-frequency
     structure expert* are softly mixed by the route prior, scaled per-channel
     by the tag prior, and spatially gated by the importance prior, producing a
     bounded residual delta added back to the features.

Every semantic-modulation layer is **zero-initialized**, so an untrained network
reduces exactly to the plain conv-attention backbone and learns to inject
guidance from there. All semantic outputs are bounded (tanh-limited gains,
clipped deltas, NaN guards) so the late-stage refinement cannot destabilize the
restoration backbone.

## Install

```bash
pip install -e .
```

Requires PyTorch ≥ 2.5 (uses `torch.nn.attention.flex_attention`) and `einops`.

## Usage

```python
import torch
from qscan import QSCAN

model = QSCAN(
    dim=64,
    pdim=16,
    kernel_size=13,
    n_blocks=5,           # 3 restoration + 2 semantic (the deepest 2 by default)
    conv_blocks=5,
    window_size=32,
    num_heads=4,
    upscaling_factor=2,
    exp_ratio=1.25,
    attn_type='Naive',    # 'Naive' | 'SDPA' | 'Flex' | 'FlashBias'
)

lr = torch.randn(1, 3, 64, 64)
sr = model(lr)                        # (1, 3, 128, 128)
sr, aux = model(lr, return_aux=True)  # aux: depth-averaged tag/route/importance logits
```

### Which blocks are semantic

By default the deepest `num_semantic_blocks=2` blocks become semantic stages and
the same indices receive the expert-fusion delta. Override explicitly with
`semantic_block_indices=[...]` and `semantic_insert_indices=[...]`.

### Attention backends

| Backend     | Notes                                                        |
|-------------|--------------------------------------------------------------|
| `Naive`     | Numerically stable reference; works everywhere.              |
| `Flex`      | Fast and memory efficient (recommended on Linux).            |
| `SDPA`      | Memory-efficient kernel; good fallback on Windows.           |
| `FlashBias` | Flash Attention with low-rank relative-position bias.        |

### Auxiliary semantic outputs

`model(x, return_aux=True)` returns `(output, aux)`, where `aux` holds the
depth-averaged `tag_logits`, `route_logits`, and `importance_logits`. During
training these are the heads supervised by the cached VLM priors; at inference
they can be inspected or ignored.

## Notes for reproduction

* `convert()` folds the geometric re-parameterization of the large-kernel filter
  into the stored weights for faster inference.
* The model registers itself in a minimal built-in `ARCH_REGISTRY`
  (`qscan.archs.registry`). If your framework has its own registry, import
  `QSCAN` directly and ignore it.

## License

MIT. See [LICENSE](LICENSE).
