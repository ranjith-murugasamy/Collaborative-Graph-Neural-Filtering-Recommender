# Reviewer comments → what this application does about them

Point-by-point mapping from both reviewer rounds in
`Enhancing Recommendation Systems_shruthik_INASS_Analysis_updated.docx.pdf` to a concrete
decision in this codebase. See `review-comments.docx` / `review-comments.txt` for the full
narrative version with real numbers; this file is the quick-reference table.

## Reviewer 1 (12 comments)

| # | Comment | Status | Where |
|---|---|---|---|
| 1 | Mathematical Formulation — define all equations, graph construction, community detection, edge reduction, training | **Implemented** | `backend/app/graph.py`, `edge_reduction.py`, `cgnf.py` transcribe Eq. (1)–(17) and Algorithms 1–3 as named, independently testable functions. |
| 2 | Experimental Setup — datasets, preprocessing, splits, negative sampling, evaluation, hyperparameters, seeds | **Implemented** | `backend/app/data.py` + run config; every run records all of these. |
| 3 | Baseline Comparison — fair protocol, include recent graph-based methods | **Implemented** | 5-way comparison (`§4.3`/`§4.4` of spec.md) under one identical protocol; LightGCN is the recent graph-based method. |
| 4 | Diversity Evaluation — Recall/NDCG aren't diversity; add coverage/novelty/ILD or reduce the claim | **Implemented** | `backend/app/diversity.py` — coverage, novelty, intra-list diversity, for every method. |
| 5 | Amazon Recall@50 inconsistency (22.67 vs 21.84) | **Resolved by replacement, not arbitration** | See `spec.md` §2 — the paper's conflicting numbers aren't adjudicated; one fresh, reproducible number replaces all of them. Amazon dataset itself is out of scope here (spec.md §7), so this specific pair is moot for this codebase but the *pattern* (don't trust either number) is documented. |
| 6 | MovieLens Recall@20 mismatches (values/method names) | **Resolved by replacement** | Same approach — `output/results/` is the one source of truth. |
| 7 | MovieLens Recall@50 inconsistency (47.05 / 49.85 / 49.86) | **Resolved by replacement** | Same approach. |
| 8 | NDCG Dataset Confusion (tables say MovieLens, text says Amazon) | **Resolved by replacement** | This application only evaluates MovieLens-1M (spec.md §7), so there is exactly one dataset label per number — the ambiguity cannot recur. |
| 9 | Improvement Percentage Error (Table 7 doesn't match the stated formula) | **Implemented** | Any improvement percentage this app reports is computed programmatically (`(new − old) / old × 100`) from the same run's numbers, not hand-typed. |
| 10 | Efficiency/Edge Reduction — prove the claim with edge count, runtime, memory | **Implemented** | `backend/app/edge_reduction.py` + `runner.py` record edge count and wall-clock before/after; see Results → Efficiency panel. |
| 11 | Ablation Study — modularity, community detection, edge clustering/reduction, GNN/NCF | **Implemented** | CGNF's own internal ablation (full / no-edge-reduction / unweighted-modularity / both disabled) — `spec.md` REQ-16. |
| 12 | Reproducibility — code, config, splits, hyperparameters, seeds, scripts, raw results | **Implemented** | Whole point of this application; every run's full config + raw per-user results exportable. `scripts/app.sh` is the single reproduction entrypoint. |

## Reviewer 2 (8 comments)

| # | Comment | Status | Where |
|---|---|---|---|
| 1 | Novelty — what's genuinely new vs. recent graph-based CF | **Partially addressed** | This app doesn't rewrite the manuscript's novelty argument (that's a writing task), but running CGNF head-to-head against LightGCN on identical data is the evidence a novelty argument would need — see Results dashboard and `review-comments.docx`. |
| 2 | Superiority/Fairness — same splits, preprocessing, eval code, candidate generation, comparable tuning | **Implemented** | Same as R1.3 — literally one evaluation code path for all 5 methods. |
| 3 | Ablation Study (modularity, partitioning, edge reduction, GNN) | **Implemented** | Duplicate of R1.11. |
| 4 | Diversity Claim | **Implemented** | Duplicate of R1.4. |
| 5 | Statistical Validation — multiple seeds, mean ± SD / CI, significance testing | **Implemented** | `backend/app/stats.py` — multi-seed mean ± SD and a Wilcoxon signed-rank test vs. the strongest baseline. Unlike the sibling project (where this was deferred), it's built here. |
| 6 | Result Consistency — fix table/figure/text mismatches | **Resolved by replacement** | Duplicate of R1.5–9's approach. |
| 7 | Generalization & Sensitivity — community settings, edge-reduction thresholds, dataset characteristics, limitations | **Implemented** | `spec.md` REQ-2 (dataset stats) + REQ-20 (interactive `(H, τ)` sensitivity sweep); limitations are `spec.md` §7, stated plainly. |
| 8 | References & Related Work — complete references, update with 2024+ studies | **Out of scope (manuscript-only)** | Literature review/reference-list work has no corresponding application feature — `spec.md` §7. |

## Issues found independently (not raised by either reviewer)

| Issue | Where addressed |
|---|---|
| No reference implementation exists at all for CGNF — only prose/equations | `spec.md` §2 — the implementation strategy (equation-by-equation transcription) is the mitigation. |
| The paper's self-reported numbers are internally inconsistent in ≥5 places, not just the ones reviewers flagged (a sixth: Table 3 vs. Table 7's MovieLens Recall@20 baseline, 10.985 vs. 10.895) | `spec.md` §2, enumerated explicitly. |
| Standard graph/GNN libraries (networkx, python-louvain, torch) all install cleanly in this environment — unlike the sibling project's genuinely broken LightFM/TensorFlow | `spec.md` §6 — the decision to hand-implement anyway is architectural (fidelity to the paper's bespoke formulation; dependency-light consistency), stated as such rather than claimed as unavoidable. |
| This project's own first community-detection implementation (a power-iteration reading of Eq. 15–17) silently collapsed to a single community regardless of target H — caught only by this app's own multi-seed validation feature (bit-identical results across seeds was the tell) | `spec.md` §2 item 3; fixed in `backend/app/graph.py` by switching to bipartite spectral co-clustering (Dhillon, 2001). |
