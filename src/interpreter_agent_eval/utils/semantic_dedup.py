"""Shared LaBSE-based semantic dedup, used by both the single-turn checklist
generator (scripts/augment_opensubs_maps.py, via consolidate_consistency_runs.py)
and the multi-turn checklist generator (pipeline/multiturn/checklist_gen.py).

Greedy, first-occurrence-priority: iterate items in order, keep an item if it
isn't a near-duplicate (cosine >= threshold) of any already-kept item.
"""
from typing import List, Sequence

import numpy as np

DEFAULT_DEDUP_THRESHOLD = 0.80

_model = None


def _load_labse():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/LaBSE")
    return _model


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    model = _load_labse()
    return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=128)


def semantic_dedup_indices(texts: Sequence[str], threshold: float = DEFAULT_DEDUP_THRESHOLD) -> List[int]:
    """Returns the surviving original indices, in input order (priority = first occurrence)."""
    if not texts:
        return []
    vecs = embed_texts(texts)
    kept_idx: List[int] = []
    kept_vecs: List[np.ndarray] = []
    for i, vec in enumerate(vecs):
        if all(float(np.dot(vec, kv)) < threshold for kv in kept_vecs):
            kept_idx.append(i)
            kept_vecs.append(vec)
    return kept_idx
