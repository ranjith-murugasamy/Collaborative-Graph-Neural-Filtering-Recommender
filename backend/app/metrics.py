"""Recall@K and NDCG@K, per-user averaged — the paper's own §4.1 states a
preference for "a more interpretable average per user rather than relying
on the global average," which this implementation actually does (unlike
the paper's own internally-inconsistent reported numbers — see
specs/spec.md §2). K defaults to {20, 50}, matching the paper's own
evaluation points, not the sibling project's {5,10,15,20}.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def _dcg_weights(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate(score_fn: Callable[[int], np.ndarray], train_user_items: Dict[int, List[int]],
             test_user_items: Dict[int, List[int]], num_items: int,
             k_list: List[int] = (20, 50)) -> Dict[str, float]:
    max_k = max(k_list)
    dcg_weights = _dcg_weights(max_k)
    sums = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg", "precision") for k in k_list}
    n_users = 0

    for user_idx, relevant in test_user_items.items():
        if not relevant:
            continue
        scores = score_fn(user_idx).copy()
        seen = train_user_items.get(user_idx, [])
        if seen:
            scores[seen] = -np.inf

        top_k = min(max_k, num_items)
        top_idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        relevant_set = set(relevant)
        hits = np.array([1 if i in relevant_set else 0 for i in top_idx], dtype=np.float64)
        ideal_hits = min(len(relevant_set), max_k)

        for k in k_list:
            hits_k = hits[:k]
            num_hits = hits_k.sum()
            sums[f"precision@{k}"] += num_hits / k
            sums[f"recall@{k}"] += num_hits / len(relevant_set)
            dcg = float(np.sum(hits_k * dcg_weights[:k]))
            idcg = float(np.sum(dcg_weights[:min(ideal_hits, k)]))
            sums[f"ndcg@{k}"] += (dcg / idcg) if idcg > 0 else 0.0
        n_users += 1

    if n_users == 0:
        return {**{k: 0.0 for k in sums}, "num_eval_users": 0}
    result = {k: v / n_users for k, v in sums.items()}
    result["num_eval_users"] = n_users
    return result
