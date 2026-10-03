"""MostPopular — matches the paper's own weakest baseline and MyMediaLite's
MostPopular.cs: score(item) = train interaction count. No learned
parameters, same score vector returned for every user (re-masked per user
by the caller against their own train-seen items, same as every other
method)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MostPopularModel:
    def __init__(self, train_matrix: sp.csr_matrix):
        binary = train_matrix.copy()
        binary.data[:] = 1.0
        self.item_counts = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)

    def score_for_user(self, user_idx: int) -> np.ndarray:
        return self.item_counts


def fit(train_matrix: sp.csr_matrix, **_ignored) -> MostPopularModel:
    return MostPopularModel(train_matrix)
