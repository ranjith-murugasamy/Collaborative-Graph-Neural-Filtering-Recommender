# CGNF Recommender

A from-scratch reproduction and honest benchmark of **CGNF** (Collaborative Graph Neural
Filtering) — the method proposed in *"Enhancing Recommendation Systems through Novel
Collaborative Graph Neural Filtering Approach"* — against four baselines (MostPopular,
Item-kNN, BPR-MF, LightGCN), built because no runnable reference implementation of CGNF exists,
only the paper's equations. Python/Flask backend, React + Tailwind CSS frontend. See
[`specs/spec.md`](specs/spec.md) for the full specification,
[`specs/reviewer-requirements.md`](specs/reviewer-requirements.md) for the point-by-point
reviewer mapping, and [`review-comments.docx`](review-comments.docx) /
[`review-comments.txt`](review-comments.txt) (same content, Word and plain-text) for the full
narrative response to all 20 Reviewer 1 / Reviewer 2 comments, with real numbers from a real
run.

## Quick start

```bash
./scripts/app.sh start      # installs deps (first run only) and starts both servers
./scripts/app.sh status
./scripts/app.sh stop
./scripts/app.sh restart
./scripts/app.sh logs backend   # tail a server's log (backend|frontend)
```

- Frontend: http://localhost:5174
- Backend API: http://127.0.0.1:5003

## What it does

Trains **5 methods** on MovieLens 1M (6,040 users, 3,883 movies, 1,000,209 ratings — in
[`input/movielens-1m/`](input/movielens-1m)) under one identical, leakage-free, per-user
held-out evaluation protocol:

- **MostPopular**, **Item-kNN**, **BPR-MF** — classic baselines, grounded in
  [MyMediaLite](http://mymedialite.net)'s canonical definitions.
- **LightGCN** — the specific recent graph-based method both reviewers asked for.
- **CGNF** — the paper's own proposed method, implemented directly from its equations (1)–(17)
  and Algorithms 1–3: weighted dual-mode bipartite community detection, edge reduction, and
  neighborhood-based scoring. Its own internal pipeline (weighted modularity, edge reduction)
  is independently toggleable for a 4-way ablation of CGNF against itself.

All 5 methods are hand-implemented in NumPy/scipy (no PyTorch, though it installs cleanly here
— see `specs/spec.md` §6 for why), with gradient-checked backward passes for every learned
component (BPR-MF, LightGCN) and a toy-graph sanity check for community detection.

The web app lets you pick or upload a dataset, configure and launch a run, watch it train live,
compare all 5 methods plus CGNF's own ablation, explore diversity (coverage/novelty/intra-list
diversity) and edge-reduction efficiency, sweep CGNF's two main hyperparameters interactively,
and run multi-seed statistical validation (mean ± SD, paired significance testing).

## Why this exists

The paper's own self-reported numbers are internally inconsistent in at least five places
(e.g. CGNF's MovieLens Recall@50 is given as 47.05, 49.85, *and* 49.86 in different passages)
— Reviewer 1 caught several of these independently. Rather than guess which number was
"right," this application reproduces CGNF and its baselines from scratch and reports exactly
one, internally-consistent number per metric, traceable to a real run. See `specs/spec.md` §2.

## Real result

CGNF (NDCG@20 = 0.0707) beats every baseline on this real run — MostPopular +115.9%, BPR-MF
+74.1%, LightGCN +38.6%, and the strongest baseline, Item-kNN, **+12.6%** — significant across
5 seeds (paired t-test, p = 4.9×10⁻⁶). Its own ablation shows *weighted modularity* is the
component that matters (+4.6%); *edge reduction* does not help accuracy at this scale (−0.2%)
while trimming 0.7% of edges. It does not win on every axis: BPR-MF has far higher catalog
coverage. See `output/README.md` for the full breakdown.

## A bug this project found in itself, not just in the paper

An initial implementation of CGNF's community detection looked correct — it ran, trained, and
produced a plausible modularity score — until this project's own multi-seed validation feature
(built to satisfy a reviewer request) produced *bit-identical* results across 5 different random
seeds. Investigation found it collapsed to a single cluster regardless of the target community
count. Fixed by switching to bipartite spectral co-clustering (Dhillon, 2001). Full account in
`specs/spec.md` §2 and `review-comments.docx`.

## Project layout

```
backend/    Flask API + models (graph.py, edge_reduction.py, cgnf.py, baselines/, metrics.py, diversity.py, stats.py, runner.py, api.py)
frontend/   React + Vite + Tailwind CSS v4 UI
input/      Sample dataset (MovieLens 1M) + cache
output/     Evidence from a real run: screenshots + exported results (see output/README.md)
scripts/    app.sh — setup/start/stop
specs/      Specification, plan, task breakdown, reviewer-requirements mapping
```

## Evidence

[`output/README.md`](output/README.md) walks through a real training run on the full dataset,
with screenshots of the live app and the exported results JSON.
