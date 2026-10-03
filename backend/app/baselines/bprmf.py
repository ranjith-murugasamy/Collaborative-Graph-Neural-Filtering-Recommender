"""BPR-MF — Bayesian Personalized Ranking matrix factorization (Rendle et
al. 2009), matching MyMediaLite's BPRMF.cs default learning rate (0.05)
and loss family: pairwise logistic loss over (user, positive item,
sampled negative item) triples. Hand-derived gradients, verified against
finite differences in backend/tests/test_models.py — same bar as every
other learned component in this project and its sibling.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..optim import Adam


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class BPRMFModel:
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_users, self.num_items, self.dim = num_users, num_items, embedding_dim
        scale = 1.0 / np.sqrt(embedding_dim)
        self.params = {
            "user": rng.normal(0, scale, size=(num_users, embedding_dim)),
            "item": rng.normal(0, scale, size=(num_items, embedding_dim)),
            "item_bias": np.zeros(num_items),
        }
        self.rng = rng

    def _sample_negatives(self, users: np.ndarray, pos_items: np.ndarray,
                           user_items_csr: sp.csr_matrix) -> np.ndarray:
        negs = np.empty_like(pos_items)
        for idx, u in enumerate(users):
            while True:
                j = self.rng.integers(0, self.num_items)
                if user_items_csr.indptr[u] == user_items_csr.indptr[u + 1] or j not in \
                        user_items_csr.indices[user_items_csr.indptr[u]:user_items_csr.indptr[u + 1]]:
                    negs[idx] = j
                    break
        return negs

    def loss_and_grad(self, users, pos_items, neg_items, l2_coef: float):
        P, Q, b = self.params["user"], self.params["item"], self.params["item_bias"]
        u_vec = P[users]
        pos_vec, neg_vec = Q[pos_items], Q[neg_items]
        pos_b, neg_b = b[pos_items], b[neg_items]

        x_uij = np.sum(u_vec * (pos_vec - neg_vec), axis=1) + (pos_b - neg_b)
        sig = _sigmoid(-x_uij)
        loss = float(np.mean(-np.log(_sigmoid(x_uij) + 1e-12)))
        if l2_coef:
            loss += l2_coef * (np.mean(u_vec ** 2) + np.mean(pos_vec ** 2) + np.mean(neg_vec ** 2))

        n = len(users)
        grad_u = -sig[:, None] * (pos_vec - neg_vec) / n + 2 * l2_coef * u_vec / n
        grad_pos = -sig[:, None] * u_vec / n + 2 * l2_coef * pos_vec / n
        grad_neg = sig[:, None] * u_vec / n + 2 * l2_coef * neg_vec / n
        grad_pos_b = -sig / n
        grad_neg_b = sig / n

        grads = {
            "user": np.zeros_like(P), "item": np.zeros_like(Q), "item_bias": np.zeros_like(b),
        }
        np.add.at(grads["user"], users, grad_u)
        np.add.at(grads["item"], pos_items, grad_pos)
        np.add.at(grads["item"], neg_items, grad_neg)
        np.add.at(grads["item_bias"], pos_items, grad_pos_b)
        np.add.at(grads["item_bias"], neg_items, grad_neg_b)
        return loss, grads

    def fit(self, train_matrix: sp.csr_matrix, *, epochs: int = 15, batch_size: int = 4096,
            lr: float = 0.05, lr_decay: float = 0.995, l2_coef: float = 1e-5,
            on_epoch=None):
        coo = train_matrix.tocoo()
        users_all, items_all = coo.row, coo.col
        n = len(users_all)
        optimizer = Adam(self.params, lr=lr)

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                users = users_all[idx]
                pos = items_all[idx]
                neg = self._sample_negatives(users, pos, train_matrix)
                loss, grads = self.loss_and_grad(users, pos, neg, l2_coef)
                optimizer.step(self.params, grads)
                epoch_loss += loss
                num_batches += 1
            optimizer.decay_lr(lr_decay)
            if on_epoch:
                on_epoch(epoch, epoch_loss / max(num_batches, 1))

    def score_for_user(self, user_idx: int) -> np.ndarray:
        return self.params["item"] @ self.params["user"][user_idx] + self.params["item_bias"]


def fit(train_matrix: sp.csr_matrix, *, embedding_dim: int = 32, epochs: int = 15,
        batch_size: int = 4096, lr: float = 0.05, lr_decay: float = 0.995,
        l2_coef: float = 1e-5, seed: int = 0, on_epoch=None, **_ignored) -> BPRMFModel:
    model = BPRMFModel(*train_matrix.shape, embedding_dim=embedding_dim, seed=seed)
    model.fit(train_matrix, epochs=epochs, batch_size=batch_size, lr=lr,
              lr_decay=lr_decay, l2_coef=l2_coef, on_epoch=on_epoch)
    return model
