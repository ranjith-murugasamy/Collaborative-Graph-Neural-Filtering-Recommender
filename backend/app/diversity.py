"""Diversity metrics — coverage, novelty, intra-list diversity. Directly
answers Reviewer 1 item 4 / Reviewer 2 item 4: "Recall and NDCG measure
accuracy, not diversity... add diversity metrics... or reduce the
diversity claim." This implementation adds them rather than walking the
claim back.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def _genre_vectors(item_genres: List[List[str]]) -> np.ndarray:
    vocab = sorted({g for genres in item_genres for g in genres})
    index = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(item_genres), max(len(vocab), 1)))
    for i, genres in enumerate(item_genres):
        for g in genres:
            mat[i, index[g]] = 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def evaluate_diversity(score_fn: Callable[[int], np.ndarray], train_user_items: Dict[int, List[int]],
                        test_user_items: Dict[int, List[int]], item_genres: List[List[str]],
                        train_item_counts: np.ndarray, top_k: int = 20) -> Dict[str, float]:
    num_items = len(item_genres)
    genre_vecs = _genre_vectors(item_genres)
    total_interactions = max(float(train_item_counts.sum()), 1.0)
    item_prob = train_item_counts / total_interactions
    item_prob = np.clip(item_prob, 1e-12, None)

    recommended_union = set()
    novelty_sum, ild_sum, n_users = 0.0, 0.0, 0

    for user_idx, relevant in test_user_items.items():
        if not relevant:
            continue
        scores = score_fn(user_idx).copy()
        seen = train_user_items.get(user_idx, [])
        if seen:
            scores[seen] = -np.inf
        k = min(top_k, num_items)
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]

        recommended_union.update(top_idx.tolist())
        novelty_sum += float(np.mean(-np.log2(item_prob[top_idx])))

        if len(top_idx) > 1:
            vecs = genre_vecs[top_idx]
            sim_matrix = vecs @ vecs.T
            iu = np.triu_indices(len(top_idx), k=1)
            pairwise_sim = sim_matrix[iu]
            ild_sum += float(np.mean(1.0 - pairwise_sim)) if len(pairwise_sim) else 0.0
        n_users += 1

    if n_users == 0:
        return {"coverage": 0.0, "novelty": 0.0, "intra_list_diversity": 0.0, "num_eval_users": 0}

    return {
        "coverage": len(recommended_union) / num_items,
        "novelty": novelty_sum / n_users,
        "intra_list_diversity": ild_sum / n_users,
        "num_eval_users": n_users,
    }
