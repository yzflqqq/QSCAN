"""QSCAN: Quality-aware Semantic Conv-Attention Network for image super-resolution.

This package provides a self-contained model definition. Only the network
architecture is included; training/evaluation pipelines are intentionally
left out so the model can be dropped into any framework.
"""

from qscan.archs.qscan_arch import QSCAN
from qscan.semantic_tags import SEMANTIC_TAGS

__all__ = ['QSCAN', 'SEMANTIC_TAGS']
__version__ = '0.1.0'
