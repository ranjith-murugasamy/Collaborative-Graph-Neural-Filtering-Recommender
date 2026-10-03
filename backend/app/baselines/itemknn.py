"""Item-kNN — cosine-similarity collaborative item neighborhoods, matching
MyMediaLite's ItemKNN.cs definition: score(u, i) = sum over u's train
items j of sim(i, j). Hand-rolled cosine similarity via scipy.sparse (no
sklearn dependency), consistent with the rest of this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class ItemKNNModel:
    train_matrix: sp.csr_matrix
    similarity: sp.csr_matrix   # (num_items, num_items), dense-ish but sparsified by top-k
    k: int

    def score_for_user(self, user_idx: int) -> np.ndarray:
        user_row = self.train_matrix[user_idx]
        scores = np.asarray((user_row @ self.similarity).todense()).ravel()
        return scores


def _row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ m


def fit(train_matrix: sp.csr_matrix, *, k: int = 40, **_ignored) -> ItemKNNModel:
    binary = train_matrix.copy().astype(np.float64)
    binary.data[:] = 1.0
    item_vectors = _row_normalize(binary.T.tocsr())          # (num_items, num_users)
    full_sim = (item_vectors @ item_vectors.T).tocsr()        # (num_items, num_items)
    full_sim.setdiag(0.0)
    full_sim.eliminate_zeros()

    # Keep only each item's top-k most-similar neighbors (standard Item-kNN
    # sparsification — MyMediaLite's ItemKNN.cs does the same).
    num_items = full_sim.shape[0]
    full_sim = full_sim.tolil()
    for i in range(num_items):
        row = full_sim.rows[i]
        data = full_sim.data[i]
        if len(row) > k:
            top = np.argpartition(-np.asarray(data), k)[:k]
            full_sim.rows[i] = [row[t] for t in top]
            full_sim.data[i] = [data[t] for t in top]
    sparsified = full_sim.tocsr()

    return ItemKNNModel(train_matrix=train_matrix, similarity=sparsified, k=k)
