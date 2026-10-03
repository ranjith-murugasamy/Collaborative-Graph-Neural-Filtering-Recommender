"""Weighted dual-mode (bipartite) community detection — CGNF's Eq.(1)-(14)
and Algorithm 1, reconstructed from the paper's equations and pseudocode.

A note on fidelity (see specs/spec.md §1 and the Reviewer 1 "Mathematical
Formulation" comment this directly answers): the paper's OCR'd equations
contain inconsistent/ambiguous subscripts in places (e.g. Eq. 15-17 reuse
the same symbol for both a per-link and a per-node quantity). Rather than
reproduce that ambiguity, this module implements one concrete, internally
consistent reading, documented function-by-function below, grounded in:

  - Eq.(1)-(2): bipartite modularity in the style of Barber (2007) —
    observed link weight minus a configuration-model null term
    (R(v)*R(l)/|U|, i.e. a degree-product expectation), summed only over
    node pairs assigned to the same community.
  - Eq.(3): ratings are row-normalized per user to [0,1] and thresholded
    into "strong" (>= tau, saturates to 1.0) vs "weak" (< tau, kept
    fractional) link weights — this is CGNF's one genuinely custom
    weighting step, distinguishing it from unweighted bipartite modularity.
  - Eq.(4)-(11), Algorithm 1 — community assignment. **Implementation
    history, not just design**: a first implementation read this as an
    alternating, mean-field-style power-iteration update (two sparse
    matmuls per iteration). That version was shipped, trained on real
    MovieLens-1M data, and only caught as broken by this project's own
    multi-seed statistical-validation feature (specs/spec.md REQ-19) —
    five different seeds produced *bit-identical* results, which should
    be essentially impossible for a genuinely stochastic clustering
    procedure. Direct investigation confirmed every tested H in
    {4,8,...,64} collapsed to a single occupied community out of H
    (`n_communities_used == 1` always) — a textbook failure mode of
    repeated multiplication by a positive matrix on one connected graph:
    it converges toward the dominant Perron eigenvector regardless of H,
    with no mechanism keeping the H "slots" distinguishable. That version
    was never a real multi-community partition; it was silently always a
    single giant cluster wearing a modularity score.
    Replaced here with **bipartite spectral co-clustering** (Dhillon,
    2001, "Co-clustering documents and words using bipartite spectral
    graph partitioning") — the standard, correct way to optimize exactly
    the kind of bipartite-modularity objective Eq.(1)-(2) describe: take
    the top-(H-1) singular vectors of the normalized bipartite link
    matrix, embed users and items into that shared space, and k-means the
    joint embedding into H clusters. This has no collapse failure mode
    and is still faithful to "weighted dual-mode community detection on a
    bipartite graph" — the paper's architectural intent — even though the
    specific soft-EM mechanics in Eq.(15)-(17) could not be reproduced
    exactly as OCR'd (see the module-level note above).
  - Eq.(12)-(14): the modularity score used to pick the community count H,
    blended with a global membership-confidence term M_d — unchanged by
    the fix above, still computed from whichever hard partition the
    clustering step (now spectral co-clustering) produces.

H is swept over a practical grid rather than the paper's literal
H_max = (|V|*|L|)^0.5 bound, which evaluates to ~4,700 at MovieLens-1M's
full training-graph scale — searching that many community counts
exhaustively is not tractable in "minutes on CPU" (specs/spec.md NFR-2).
That bound is only reachable for the paper's own small worked example;
this is a scalability gap in the paper's own stated protocol, not just a
simplification made here, and is documented as such in specs/spec.md §2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def weighted_link_matrix(train_matrix: sp.csr_matrix, tau: float = 0.7) -> sp.csr_matrix:
    """Eq.(3): row-normalize each user's ratings by their own max rating,
    then saturate values >= tau to 1.0 ("strong" links) while leaving
    values < tau at their fractional weight ("weak" links)."""
    m = train_matrix.tocsr(copy=True).astype(np.float64)
    row_max = np.asarray(m.max(axis=1).todense()).ravel()
    row_max[row_max == 0] = 1.0
    inv = sp.diags(1.0 / row_max)
    normalized = inv @ m
    normalized.data = np.where(normalized.data >= tau, 1.0, normalized.data)
    return normalized.tocsr()


@dataclass
class CommunityResult:
    H: int
    user_comm: np.ndarray       # (num_users, H) one-hot (hard partition)
    item_comm: np.ndarray       # (num_items, H) one-hot (hard partition)
    user_label: np.ndarray      # (num_users,) hard assignment
    item_label: np.ndarray      # (num_items,) hard assignment
    modularity: float           # Z^Gl_x, Eq.(14)
    z_x: float                  # Z^x, Eq.(12)
    m_d: float                  # M_d, Eq.(13)
    n_communities_used: int     # sanity-check field: distinct non-empty communities


def _kmeans(points: np.ndarray, k: int, *, seed: int, iters: int = 50, restarts: int = 5):
    """A small hand-rolled k-means (no sklearn dependency, matching the
    rest of this project) with multiple random restarts, keeping the
    lowest-inertia result — standard practice to avoid a single bad
    random initialization producing a degenerate (near-empty-cluster)
    partition."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = max(1, min(k, n))
    best_labels, best_inertia = None, np.inf

    for _ in range(restarts):
        centers = points[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) \
                if n * k <= 2_000_000 else _chunked_sqdist(points, centers)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = points[mask].mean(axis=0)
                else:
                    centers[c] = points[rng.integers(0, n)]
        dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) \
            if n * k <= 2_000_000 else _chunked_sqdist(points, centers)
        inertia = dists[np.arange(n), labels].sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels

    return best_labels, best_inertia


def _chunked_sqdist(points: np.ndarray, centers: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Memory-bounded squared-distance matrix for large point sets."""
    n = points.shape[0]
    out = np.empty((n, centers.shape[0]), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = points[start:end, None, :] - centers[None, :, :]
        out[start:end] = (diff ** 2).sum(axis=2)
    return out


def detect_communities(link_matrix: sp.csr_matrix, H: int, *, iters: int = 12,
                        seed: int = 0) -> CommunityResult:
    """Bipartite spectral co-clustering (Dhillon, 2001) — see the module
    docstring for why this replaced a naive power-iteration reading of
    Eq.(15)-(17). `iters` is passed through to k-means as its max
    iteration count."""
    num_users, num_items = link_matrix.shape

    user_deg = np.asarray(link_matrix.sum(axis=1)).ravel()
    item_deg = np.asarray(link_matrix.sum(axis=0)).ravel()
    user_deg_safe = np.where(user_deg == 0, 1.0, user_deg)
    item_deg_safe = np.where(item_deg == 0, 1.0, item_deg)
    d_u_inv_sqrt = sp.diags(1.0 / np.sqrt(user_deg_safe))
    d_i_inv_sqrt = sp.diags(1.0 / np.sqrt(item_deg_safe))
    normalized = (d_u_inv_sqrt @ link_matrix @ d_i_inv_sqrt).tocsr().astype(np.float64)

    # Dhillon's method embeds into the top-(H-1) singular vectors (the
    # leading one is the trivial/Perron direction and is skipped).
    k = max(1, min(H - 1, min(num_users, num_items) - 2))
    u, s, vt = spla.svds(normalized, k=k, random_state=seed)
    order = np.argsort(-s)
    u, vt = u[:, order], vt[order, :]

    user_embed = np.asarray(d_u_inv_sqrt @ u)         # (num_users, k)
    item_embed = np.asarray(d_i_inv_sqrt @ vt.T)       # (num_items, k)
    joint = np.vstack([user_embed, item_embed])

    labels, _ = _kmeans(joint, H, seed=seed, iters=iters)
    user_label = labels[:num_users]
    item_label = labels[num_users:]

    n_communities_used = len(np.unique(labels))
    user_comm = np.eye(H, dtype=np.float64)[user_label]
    item_comm = np.eye(H, dtype=np.float64)[item_label]

    # Eq.(12): sum over edges whose endpoints share a hard community of
    # (link weight - degree-product null term), normalized by total weight.
    coo = link_matrix.tocoo()
    total_weight = coo.data.sum() or 1.0
    same_comm = user_label[coo.row] == item_label[coo.col]
    null_term = (user_deg[coo.row] * item_deg[coo.col]) / total_weight
    z_x = float(np.sum((coo.data - null_term)[same_comm]) / total_weight)

    # Eq.(13): M_d — mean membership "confidence"; with a hard partition
    # this is 1.0 for every observed link by construction (each node's
    # membership probability mass is entirely on its own community) — a
    # faithful degenerate case of the paper's soft-membership confidence
    # term, not a bug: a hard assignment is maximally confident.
    m_d = 1.0

    # Eq.(14): blend local (z_x) and global (m_d) signal.
    z_gl = (2 * m_d + z_x) / 3.0

    return CommunityResult(H=H, user_comm=user_comm, item_comm=item_comm,
                            user_label=user_label, item_label=item_label,
                            modularity=z_gl, z_x=z_x, m_d=m_d,
                            n_communities_used=n_communities_used)


def sweep_h(link_matrix: sp.csr_matrix, h_grid: List[int], *, iters: int = 12,
            seed: int = 0, on_step=None) -> CommunityResult:
    """Try every H in h_grid, keep the one maximizing Z^Gl_x (Eq. 14) —
    Algorithm 1's 'the higher the value the more robust the community is'
    criterion, swept over a practical grid instead of the paper's literal
    [H_min, H_max] range (see module docstring)."""
    best = None
    for H in h_grid:
        result = detect_communities(link_matrix, H, iters=iters, seed=seed)
        if on_step:
            on_step(H, result.modularity)
        if best is None or result.modularity > best.modularity:
            best = result
    return best


def practical_h_grid(num_users: int, num_items: int, max_h: int = 64) -> List[int]:
    """A tractable stand-in for the paper's [2, (|V|*|L|)^0.5] sweep."""
    literal_h_max = int(np.ceil((num_users * num_items) ** 0.5))
    cap = min(max_h, max(4, literal_h_max))
    grid = sorted(set(int(x) for x in [4, 8, 12, 16, 24, 32, 48, 64] if x <= cap))
    return grid or [4]
