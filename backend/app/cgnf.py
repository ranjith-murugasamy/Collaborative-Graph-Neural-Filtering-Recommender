"""CGNF end to end — Algorithm 3 (recommendation generation) built on top
of graph.py's community detection and edge_reduction.py's redundancy
removal. Every sub-step is independently toggleable so the ablation study
(specs/spec.md REQ-16 / Reviewer 1 item 11 / Reviewer 2 item 3) reuses this
exact function with different flags, not a separate "ablation mode."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from . import edge_reduction as er
from . import graph as graph_mod


@dataclass
class CGNFModel:
    num_users: int
    num_items: int
    user_label: np.ndarray
    item_label: np.ndarray
    H: int
    modularity: float
    edges_before: int
    edges_after: int
    neighbor_matrix: sp.csr_matrix       # reduced (or full) link matrix used for scoring
    raw_train_matrix: sp.csr_matrix      # original ratings, for score weighting


def fit(train_matrix: sp.csr_matrix, *, tau: float = 0.7, h_grid=None,
        weighted_modularity: bool = True, use_edge_reduction: bool = True,
        iters: int = 12, seed: int = 0, on_h_step=None) -> CGNFModel:
    """Train CGNF. Ablation toggles:
      - weighted_modularity=False: skip Eq.(3)'s strong/weak weighting and
        use raw binary interactions instead (tests whether the custom
        weighting in Eq.(3)-(4) actually matters, vs. plain bipartite
        modularity on an unweighted graph).
      - use_edge_reduction=False: skip Algorithm 2 entirely (tests whether
        edge reduction changes recommendation quality, not just speed).
    """
    num_users, num_items = train_matrix.shape
    if h_grid is None:
        h_grid = graph_mod.practical_h_grid(num_users, num_items)

    link_matrix = (graph_mod.weighted_link_matrix(train_matrix, tau=tau)
                   if weighted_modularity else _binary(train_matrix))

    best = graph_mod.sweep_h(link_matrix, h_grid, iters=iters, seed=seed, on_step=on_h_step)

    if use_edge_reduction:
        er_result = er.reduce_edges(link_matrix, best.user_label)
        neighbor_matrix = er_result.reduced_matrix
        edges_before, edges_after = er_result.edges_before, er_result.edges_after
    else:
        edges_before = int(link_matrix.nnz)
        neighbor_matrix = link_matrix
        edges_after = edges_before

    return CGNFModel(
        num_users=num_users, num_items=num_items,
        user_label=best.user_label, item_label=best.item_label,
        H=best.H, modularity=best.modularity,
        edges_before=edges_before, edges_after=edges_after,
        neighbor_matrix=neighbor_matrix, raw_train_matrix=train_matrix,
    )


def _binary(m: sp.csr_matrix) -> sp.csr_matrix:
    b = m.copy().astype(np.float64)
    b.data[:] = 1.0
    return b


def score_for_user(model: CGNFModel, user_idx: int, top_k_neighbors: int = 50) -> np.ndarray:
    """Algorithm 3: locate the target user's community, find similarity-
    weighted neighbors within it (restricted to the edge-reduced set when
    that ablation arm is on), aggregate their ratings into a score per
    item — classic user-based CF prediction, scoped to one community
    instead of the whole user base."""
    community = model.user_label[user_idx]
    members = np.where(model.user_label == community)[0]
    members = members[members != user_idx]
    if members.size == 0:
        return np.zeros(model.num_items)

    target_row = model.neighbor_matrix[user_idx]
    if target_row.nnz == 0:
        target_row = model.raw_train_matrix[user_idx]
    neighbor_rows = model.neighbor_matrix[members]

    target_norm = np.sqrt(target_row.multiply(target_row).sum()) or 1.0
    neighbor_norms = np.sqrt(np.asarray(neighbor_rows.multiply(neighbor_rows).sum(axis=1))).ravel()
    neighbor_norms[neighbor_norms == 0] = 1.0
    sims = np.asarray((neighbor_rows @ target_row.T).todense()).ravel() / (neighbor_norms * target_norm)

    if top_k_neighbors and members.size > top_k_neighbors:
        top = np.argpartition(-sims, top_k_neighbors - 1)[:top_k_neighbors]
        members, sims = members[top], sims[top]

    sims = np.clip(sims, 0, None)
    weight_sum = sims.sum()
    if weight_sum <= 1e-12:
        return np.zeros(model.num_items)

    neighbor_ratings = model.raw_train_matrix[members]
    scores = np.asarray((sp.diags(sims) @ neighbor_ratings).sum(axis=0)).ravel() / weight_sum
    return scores
