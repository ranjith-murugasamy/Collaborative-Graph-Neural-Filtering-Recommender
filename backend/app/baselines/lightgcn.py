"""LightGCN (He et al., SIGIR 2020) — the specific "modern collaborative
approach" both reviewers name explicitly. Embeddings are propagated
through K layers of the symmetrically-normalized bipartite adjacency
(no feature transformation, no nonlinearity — LightGCN's whole point),
the final embedding is the mean of all K+1 layers, and training uses the
same BPR pairwise loss as bprmf.py.

Implemented in NumPy, not PyTorch, even though PyTorch installs cleanly
in this environment (specs/spec.md §6) — every learned component in this
project is NumPy with a hand-derived, gradient-checked backward pass, and
LightGCN's computation (linear propagation + a BPR loss) has no real need
for autodiff. The backward pass here manually unrolls the K propagation
layers in reverse, which is straightforward because propagation is linear.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..optim import Adam


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def normalized_adjacency(train_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """A_tilde = D_u^{-1/2} R D_i^{-1/2} — the user->item half of LightGCN's
    symmetric normalization; its transpose is the item->user half."""
    user_deg = np.asarray(train_matrix.sum(axis=1)).ravel()
    item_deg = np.asarray(train_matrix.sum(axis=0)).ravel()
    user_deg[user_deg == 0] = 1.0
    item_deg[item_deg == 0] = 1.0
    d_u_inv_sqrt = sp.diags(1.0 / np.sqrt(user_deg))
    d_i_inv_sqrt = sp.diags(1.0 / np.sqrt(item_deg))
    return (d_u_inv_sqrt @ train_matrix @ d_i_inv_sqrt).tocsr()


class LightGCNModel:
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32,
                 num_layers: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_users, self.num_items = num_users, num_items
        self.dim, self.K = embedding_dim, num_layers
        scale = 1.0 / np.sqrt(embedding_dim)
        self.params = {
            "user0": rng.normal(0, scale, size=(num_users, embedding_dim)),
            "item0": rng.normal(0, scale, size=(num_items, embedding_dim)),
        }
        self.rng = rng
        self.adj = None  # set in fit()

    def _propagate(self, user0, item0):
        """Returns (U_layers, I_layers), lists of length K+1."""
        U, I = [user0], [item0]
        for _ in range(self.K):
            U.append(self.adj @ I[-1])
            I.append(self.adj.T @ U[-2])
        return U, I

    def _final_embeddings(self, user0, item0):
        U, I = self._propagate(user0, item0)
        u_final = sum(U) / (self.K + 1)
        i_final = sum(I) / (self.K + 1)
        return u_final, i_final, U, I

    def score_matrix_for_user(self, user_idx: int, u_final=None, i_final=None) -> np.ndarray:
        if u_final is None:
            u_final, i_final, _, _ = self._final_embeddings(self.params["user0"], self.params["item0"])
        return i_final @ u_final[user_idx]

    def _sample_negatives(self, users: np.ndarray, train_matrix: sp.csr_matrix) -> np.ndarray:
        negs = np.empty_like(users)
        for idx, u in enumerate(users):
            start, end = train_matrix.indptr[u], train_matrix.indptr[u + 1]
            seen = train_matrix.indices[start:end]
            while True:
                j = self.rng.integers(0, self.num_items)
                if j not in seen:
                    negs[idx] = j
                    break
        return negs

    def _loss_and_grad(self, users, pos_items, neg_items, l2_coef: float):
        user0, item0 = self.params["user0"], self.params["item0"]
        u_final, i_final, U, I = self._final_embeddings(user0, item0)

        u_vec = u_final[users]
        pos_vec, neg_vec = i_final[pos_items], i_final[neg_items]
        x = np.sum(u_vec * (pos_vec - neg_vec), axis=1)
        sig = _sigmoid(-x)
        n = len(users)
        loss = float(np.mean(-np.log(_sigmoid(x) + 1e-12)))
        if l2_coef:
            loss += l2_coef * (np.mean(u_vec ** 2) + np.mean(pos_vec ** 2) + np.mean(neg_vec ** 2))

        d_u_final = np.zeros_like(u_final)
        d_i_final = np.zeros_like(i_final)
        grad_u = -sig[:, None] * (pos_vec - neg_vec) / n + 2 * l2_coef * u_vec / n
        grad_pos = -sig[:, None] * u_vec / n + 2 * l2_coef * pos_vec / n
        grad_neg = sig[:, None] * u_vec / n + 2 * l2_coef * neg_vec / n
        np.add.at(d_u_final, users, grad_u)
        np.add.at(d_i_final, pos_items, grad_pos)
        np.add.at(d_i_final, neg_items, grad_neg)

        # d(final)/d(layer k) = 1/(K+1) for every layer (mean pooling).
        dU = [d_u_final / (self.K + 1) for _ in range(self.K + 1)]
        dI = [d_i_final / (self.K + 1) for _ in range(self.K + 1)]

        # Backprop through the linear propagation, layers K..1:
        #   U[k] = adj @ I[k-1]      => dI[k-1] += adj.T @ dU[k]
        #   I[k] = adj.T @ U[k-1]    => dU[k-1] += adj @ dI[k]
        for k in range(self.K, 0, -1):
            dI[k - 1] = dI[k - 1] + np.asarray(self.adj.T @ dU[k])
            dU[k - 1] = dU[k - 1] + np.asarray(self.adj @ dI[k])

        grads = {"user0": dU[0], "item0": dI[0]}
        return loss, grads

    def fit(self, train_matrix: sp.csr_matrix, *, epochs: int = 15, batch_size: int = 4096,
            lr: float = 0.01, lr_decay: float = 0.995, l2_coef: float = 1e-5, on_epoch=None):
        self.adj = normalized_adjacency(train_matrix)
        coo = train_matrix.tocoo()
        users_all, items_all = coo.row, coo.col
        n = len(users_all)
        optimizer = Adam(self.params, lr=lr)

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            epoch_loss, num_batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                users = users_all[idx]
                pos = items_all[idx]
                neg = self._sample_negatives(users, train_matrix)
                loss, grads = self._loss_and_grad(users, pos, neg, l2_coef)
                optimizer.step(self.params, grads)
                epoch_loss += loss
                num_batches += 1
            optimizer.decay_lr(lr_decay)
            if on_epoch:
                on_epoch(epoch, epoch_loss / max(num_batches, 1))

        self._u_final, self._i_final, _, _ = self._final_embeddings(
            self.params["user0"], self.params["item0"])

    def score_for_user(self, user_idx: int) -> np.ndarray:
        return self._i_final @ self._u_final[user_idx]


def fit(train_matrix: sp.csr_matrix, *, embedding_dim: int = 32, num_layers: int = 3,
        epochs: int = 15, batch_size: int = 4096, lr: float = 0.01, lr_decay: float = 0.995,
        l2_coef: float = 1e-5, seed: int = 0, on_epoch=None, **_ignored) -> LightGCNModel:
    model = LightGCNModel(*train_matrix.shape, embedding_dim=embedding_dim,
                           num_layers=num_layers, seed=seed)
    model.fit(train_matrix, epochs=epochs, batch_size=batch_size, lr=lr,
              lr_decay=lr_decay, l2_coef=l2_coef, on_epoch=on_epoch)
    return model
