# Task breakdown

Checklist form of `plan.md`, grouped to match `reviewer-requirements.md`'s traceability.

## Setup
- [x] Reuse MovieLens 1M dataset in `input/`
- [x] Project scaffold: `specs/`, `backend/`, `frontend/`, `scripts/`, `output/`
- [x] Verify dependency landscape (networkx/python-louvain/torch install cleanly; decided NumPy anyway — spec.md §6)

## Backend — data & core algorithm
- [x] `data.py`: parse `.dat` files, leave-k-out split, bipartite matrix, dataset stats
- [x] `graph.py`: weighted dual-mode modularity (Eq. 1–14), community detection (Algorithm 1) — **rewritten mid-project**: an initial power-iteration reading collapsed to a single community regardless of H (caught via multi-seed validation's bit-identical-across-seeds symptom); replaced with bipartite spectral co-clustering (Dhillon, 2001), verified to produce genuine H-way partitions
- [x] `edge_reduction.py`: Eq. 15–17 support, Algorithm 2 (vectorized, not a Python double loop)
- [x] `cgnf.py`: Algorithm 3 neighborhood scoring, ties graph.py + edge_reduction.py together, toggleable sub-steps for ablation
- [x] Toy-graph sanity check: Algorithm 1 recovers an obvious 2-community structure — passes with both the old and fixed implementation (didn't catch the collapse bug; only multi-seed validation on the real, non-toy graph did)

## Backend — baselines
- [x] `baselines/mostpop.py`
- [x] `baselines/itemknn.py`
- [x] `baselines/bprmf.py` (gradient-checked, max abs error 6.5e-12)
- [x] `baselines/lightgcn.py` (gradient-checked, max abs error 1.1e-11)

## Backend — evaluation
- [x] `metrics.py`: Recall@{20,50}, NDCG@{20,50}, per-user averaged
- [x] `diversity.py`: coverage, novelty, intra-list diversity
- [x] `stats.py`: multi-seed mean ± SD, paired t-test + Wilcoxon signed-rank
- [x] Efficiency tracking: edge count + wall-clock before/after reduction

## Backend — service
- [x] `runner.py`: orchestration, live state, sensitivity-grid caching, multi-seed on-demand job
- [x] `api.py`: REST + SSE routes per `plan.md`
- [x] Smoke test on a small sampled subset before the full run

## Frontend
- [x] Project scaffold (Vite + React + Tailwind v4) — **third distinct visual identity**: light-first indigo/cyan "network lab" theme, monospace display font, dot-grid texture, hexagonal node badges, top underline-tab nav (distinct from both sibling projects' layouts and palettes)
- [x] Datasets page
- [x] Train page
- [x] Results dashboard: 5-way comparison, CGNF ablation, diversity, efficiency panels
- [x] Sensitivity sweep page (community count × edge-reduction threshold)
- [x] Statistical validation page (multi-seed)
- [x] Header/nav, color system, dark mode, responsive layout

## Tooling
- [x] `scripts/app.sh`: setup / start / stop / restart / status / logs (single script)

## Run & evidence
- [x] Full run on real MovieLens 1M (all 5 methods + 4-way CGNF ablation + diversity + efficiency), re-run after the community-detection fix
- [x] Multi-seed statistical validation run for real (5 seeds, CGNF vs. strongest baseline)
- [x] Capture dashboard screenshots + exported results into `output/`
- [x] `review-comments.docx`: point-by-point response to all 20 reviewer comments + the independently-found issues (including the self-caught community-detection bug)
- [x] Update `specs/tasks.md` checkboxes to reflect final state
