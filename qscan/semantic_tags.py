"""Semantic tag vocabulary used by the network's semantic-conditioning paths.

These tags describe the dominant low-level content of a patch and drive the
lightweight semantic modulation inside the model. The vocabulary is a plain
tuple so the model can be configured with ``num_tags=len(SEMANTIC_TAGS)``.
"""

from __future__ import annotations

SEMANTIC_TAGS = (
    'text',
    'thin_edge',
    'line_art',
    'repetitive_pattern',
    'brick_window_grid',
    'hair_fur',
    'plant_leaf',
    'natural_texture',
    'face',
    'document_sign',
    'anime_manga',
    'flat_smooth_region',
)

__all__ = ['SEMANTIC_TAGS']
