"""Gradient checks (BPR-MF, LightGCN) and a toy community-detection sanity
check — same bar as the sibling project's test_models.py: trust nothing
hand-derived until it's been checked against finite differences or an
obviously-correct small example.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from app import graph as graph_mod
from app.baselines.bprmf import BPRMFModel
from app.baselines.lightgcn import LightGCNModel, normalized_adjacency


def check_grads(params, grads, loss_fn, eps=1e-5, samples=6, seed=0):
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for name, arr in params.items():
        flat = arr.reshape(-1)
        g_flat = grads[name].reshape(-1)
        idxs = rng.choice(flat.size, size=min(samples, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = loss_fn()
            flat[idx] = orig - eps
            loss_minus = loss_fn()
            flat[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = g_flat[idx]
            err = abs(numeric - analytic)
            max_err = max(max_err, err)
    return max_err


def check_bprmf():
    rng = np.random.default_rng(0)
    num_users, num_items = 12, 15
    density = 0.3
    mat = (rng.random((num_users, num_items)) < density).astype(np.float64) * rng.integers(1, 6, (num_users, num_items))
    train_matrix = sp.csr_matrix(mat)

    model = BPRMFModel(num_users, num_items, embedding_dim=4, seed=1)
    coo = train_matrix.tocoo()
    users, pos = coo.row[:8], coo.col[:8]
    neg = np.array([rng.integers(0, num_items) for _ in users])

    rng_state = model.rng.bit_generator.state

    def loss_fn():
        model.rng.bit_generator.state = rng_state
        loss, _ = model.loss_and_grad(users, pos, neg, l2_coef=0.0)
        return loss

    model.rng.bit_generator.state = rng_state
    loss, grads = model.loss_and_grad(users, pos, neg, l2_coef=0.0)
    err = check_grads(model.params, grads, loss_fn)
    print(f"[bprmf] max abs grad error: {err:.3e}")
    assert err < 1e-4, f"bprmf gradient check failed: {err}"


def check_lightgcn():
    rng = np.random.default_rng(2)
    num_users, num_items = 10, 12
    mat = (rng.random((num_users, num_items)) < 0.35).astype(np.float64) * rng.integers(1, 6, (num_users, num_items))
    train_matrix = sp.csr_matrix(mat)

    model = LightGCNModel(num_users, num_items, embedding_dim=4, num_layers=2, seed=3)
    model.adj = normalized_adjacency(train_matrix)
    coo = train_matrix.tocoo()
    users, pos = coo.row[:6], coo.col[:6]
    neg = np.array([rng.integers(0, num_items) for _ in users])

    def loss_fn():
        loss, _ = model._loss_and_grad(users, pos, neg, l2_coef=0.0)
        return loss

    loss, grads = model._loss_and_grad(users, pos, neg, l2_coef=0.0)
    err = check_grads(model.params, grads, loss_fn)
    print(f"[lightgcn] max abs grad error: {err:.3e}")
    assert err < 1e-4, f"lightgcn gradient check failed: {err}"


def check_toy_community_detection():
    """4 users, 4 items, two obvious disjoint 2x2 blocks — community
    detection with H=2 should separate them perfectly."""
    mat = np.array([
        [5, 4, 0, 0],
        [4, 5, 0, 0],
        [0, 0, 5, 4],
        [0, 0, 4, 5],
    ], dtype=np.float64)
    train_matrix = sp.csr_matrix(mat)
    link_matrix = graph_mod.weighted_link_matrix(train_matrix, tau=0.7)
    result = graph_mod.detect_communities(link_matrix, H=2, iters=20, seed=0)

    same = result.user_label[0] == result.user_label[1]
    other = result.user_label[2] == result.user_label[3]
    cross = result.user_label[0] != result.user_label[2]
    print(f"[toy] user_labels={result.user_label.tolist()} modularity={result.modularity:.4f}")
    assert same and other and cross, "toy community detection failed to separate the two obvious blocks"


if __name__ == "__main__":
    check_bprmf()
    check_lightgcn()
    check_toy_community_detection()
    print("All checks passed.")
