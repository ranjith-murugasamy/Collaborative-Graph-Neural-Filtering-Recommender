"""Edge reduction — Algorithm 2: within each community, a user whose
rated-item set is a subset of another user's rated-item set in the same
community contributes no information a collaborative-filtering neighbor
search wouldn't already get from the superset user, so their links are
dropped from the *reduced* edge set used for neighbor search (they remain
fully queryable as a target user in cgnf.py — only their contribution as a
*neighbor signal* is pruned).

Implemented as a dense boolean subset check per community (tractable since
communities are a small fraction of the full user base — see
detect_communities in graph.py), not a literal pairwise loop over all
users against the whole graph, which the paper's RL-wide framing would
otherwise imply and which does not scale to MovieLens-1M.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import scipy.sparse as sp


@dataclass
class EdgeReductionResult:
    redundant_users: np.ndarray       # boolean, shape (num_users,)
    edges_before: int
    edges_after: int
    reduced_matrix: sp.csr_matrix     # link_matrix with redundant users' rows zeroed


def reduce_edges(link_matrix: sp.csr_matrix, user_label: np.ndarray) -> EdgeReductionResult:
    binary = link_matrix.copy()
    binary.data[:] = 1.0
    num_users = binary.shape[0]
    redundant = np.zeros(num_users, dtype=bool)

    for community in np.unique(user_label):
        members = np.where(user_label == community)[0]
        n_c = members.size
        if n_c < 2:
            continue
        sub = binary[members]                     # (n_c, num_items) boolean rows
        row_nnz = np.asarray(sub.sum(axis=1)).ravel()
        co = np.asarray((sub @ sub.T).todense())   # (n_c, n_c) co-occurrence counts

        # user a is redundant if some other user b (with at least as many
        # items) has a's entire item set as a subset of b's — vectorized
        # over the whole community at once rather than a Python double loop,
        # which does not scale to communities of a few hundred users,
        # especially re-run across every cell of the sensitivity sweep.
        has_items = row_nnz > 0
        is_subset = (co == row_nnz[:, None]) & (row_nnz[None, :] >= row_nnz[:, None])
        np.fill_diagonal(is_subset, False)
        community_redundant = has_items & is_subset.any(axis=1)
        redundant[members[community_redundant]] = True

    edges_before = int(binary.nnz)
    reduced = link_matrix.tolil(copy=True)
    reduced[redundant, :] = 0.0
    reduced = reduced.tocsr()
    reduced.eliminate_zeros()
    edges_after = int(reduced.nnz)

    return EdgeReductionResult(redundant_users=redundant, edges_before=edges_before,
                                edges_after=edges_after, reduced_matrix=reduced)
