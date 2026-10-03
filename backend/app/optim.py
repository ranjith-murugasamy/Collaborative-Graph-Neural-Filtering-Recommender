"""Generic Adam optimizer over a dict of named NumPy arrays — identical
pattern to the sibling hybrid-movie-recommender project, reused as-is for
BPR-MF and LightGCN here."""
from __future__ import annotations

from typing import Dict

import numpy as np


class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.01,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def decay_lr(self, factor: float):
        self.lr *= factor
