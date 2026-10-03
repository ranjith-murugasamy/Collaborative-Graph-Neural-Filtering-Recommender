# CGNF Recommender — Technical Plan

Companion to `spec.md` (§4's REQ-ids are referenced throughout) and `reviewer-requirements.md`
(comment-by-comment mapping).

## 1. Directory layout

```
cgnf-recommender/
├── backend/
│   ├── app/
│   │   ├── data.py           # MovieLens parsing, bipartite matrix, leave-k-out split
│   │   ├── graph.py          # Eq.(1)-(14): weighted dual-mode modularity, community detection (Algorithm 1)
│   │   ├── edge_reduction.py # Eq.(15)-(17) support + Algorithm 2
│   │   ├── cgnf.py           # Algorithm 3: neighborhood scoring, ties graph.py + edge_reduction.py together
│   │   ├── baselines/
│   │   │   ├── mostpop.py
│   │   │   ├── itemknn.py
│   │   │   ├── bprmf.py
│   │   │   └── lightgcn.py
│   │   ├── metrics.py        # Recall@K, NDCG@K, per-user averaged
│   │   ├── diversity.py      # coverage, novelty, intra-list diversity
│   │   ├── stats.py          # multi-seed aggregation, Wilcoxon signed-rank
│   │   ├── runner.py         # orchestration: train all 5 methods + ablation + sensitivity
│   │   ├── custom_store.py   # uploaded-dataset storage (mirrors sibling project)
│   │   └── api.py            # Flask REST + SSE
│   ├── tests/test_models.py  # gradient checks (BPR-MF, LightGCN) + modularity sanity checks
│   └── requirements.txt
├── frontend/                 # React + Vite + Tailwind v4 — distinct visual identity (see below)
├── input/movielens-1m/
├── output/{screenshots,results}/
├── scripts/app.sh
└── specs/{spec.md,plan.md,tasks.md,reviewer-requirements.md}
```

## 2. Data flow

1. Parse `movies.dat`/`ratings.dat` → per-user leave-k-out split (reusing the sibling project's
   proven pattern: last `k` interactions per user by timestamp → test; rest → train; users with
   ≤ `min_interactions` stay entirely in train).
2. Build the bipartite incidence matrix `D` (users × items, scipy.sparse) from **train** data
   only.
3. **Community detection** (`graph.py`): iterate Algorithm 1 — for each link `U_v,l`, compute
   `p(U_h|U_v,l)` via Eq. (15)–(17) using the current `pred_L`/`pred_V` estimates (Eq. 10–11),
   assign to the highest-probability community, recompute `Z^x` (Eq. 12) and the blended
   `Z^Gl_x` (Eq. 14); repeat for `H` from `H_min=2` to `H_max=⌈(|V|·|L|)^0.5⌉`, keep the `H`
   that maximizes `Z^Gl_x`.
4. **Edge reduction** (`edge_reduction.py`, Algorithm 2): within each resulting community,
   drop a link `U_c` if its endpoint's link set is already a subset of another member's —
   record edge count before/after for REQ-18.
5. **Recommendation** (`cgnf.py`, Algorithm 3): for a target user, locate their community,
   rank unrated items by similarity-weighted neighbor agreement within that community.
6. Baselines (`baselines/`) train independently on the same train split; all five methods are
   evaluated through the same `metrics.py`/`diversity.py` functions against the same held-out
   test split — no method gets a different evaluation code path.

## 3. Why community detection isn't "instant" the way fusion weights were

The sibling project's fusion-weight sliders are instant because the three component score
matrices are precomputed once and the fusion step is a cheap weighted sum over a fixed matrix.
CGNF's hyperparameters (community count, edge-reduction threshold) change the *community
assignment itself* — a different, non-trivial computation, not a reweighting of a fixed
output. The sensitivity sweep (REQ-20) therefore:

- Precomputes a small grid (e.g. 5 community-count values × 4 threshold values = 20 cells) in
  the background when a run starts, cached by `(H, τ)` key.
- The UI reads from that cache instantly for any point already computed, and shows a
  "computing…" state (with the same SSE log-streaming pattern as training) for a cell outside
  the precomputed grid, computed on demand.

## 4. Baseline math (grounded in MyMediaLite's definitions, reimplemented in NumPy)

- **MostPopular** — score(i) = train interaction count for item `i`. No learned parameters.
- **Item-kNN** — cosine similarity between items' train-interaction column vectors; score for
  user *u*, item *i* = similarity-weighted sum over *u*'s train items.
- **BPR-MF** — user/item embeddings, pairwise logistic loss over (user, positive, sampled
  negative) triples, SGD/Adam, default `lr=0.05` matching MyMediaLite's `BPRMF.cs`.
- **LightGCN** — `K`-layer propagation `E^(k+1) = D_u^(-1/2) A D_i^(-1/2) E^(k)` over the
  normalized bipartite adjacency `A`; final embedding = mean of `E^(0)..E^(K)`; same BPR loss
  as above. `K` default 3, per He et al. (2020).

All four are gradient-checked (where they have learned parameters) in `backend/tests/`, same
bar as the sibling project (max abs error target ~1e-9).

## 5. API surface

| Route | Purpose |
|---|---|
| `GET /api/health` | liveness |
| `GET /api/config/defaults` | hyperparameter defaults for all 5 methods + ablation/sensitivity grid definition |
| `GET /api/datasets` / `POST /api/datasets/custom` | dataset listing / upload |
| `POST /api/runs` | start a run (train all 5 methods + ablation combos) |
| `GET /api/runs` / `GET /api/runs/<id>` / `GET /api/runs/<id>/logs` / `GET /api/runs/<id>/stream` | run listing/detail/logs (SSE) |
| `GET /api/runs/<id>/sensitivity?H=&tau=` | sensitivity-grid cell (cached or computed on demand) |
| `GET /api/runs/<id>/diversity` | diversity table for all 5 methods |
| `GET /api/runs/<id>/efficiency` | edge-count/runtime before/after reduction |
| `POST /api/runs/<id>/multiseed` | kick off REQ-19's multi-seed statistical validation |
| `GET /api/runs/<id>/recommend?user_id=&method=` | per-user Top-N, any of the 5 methods |

## 6. Frontend pages

Datasets · Train · Results (five-way comparison + CGNF ablation + diversity + efficiency) ·
Sensitivity (the `(H, τ)` grid) · Statistical Validation (multi-seed). Visual identity is
deliberately distinct from both sibling projects — see `frontend/README.md` once scaffolded
for the specific design-language decisions (this is the third distinct UI in this workspace;
reusing either prior theme was explicitly ruled out).

## 7. Validation loop

1. Unit-level: gradient checks (BPR-MF, LightGCN) + a hand-computable modularity sanity check
   (a toy 4-user/4-item graph with an obvious 2-community structure; assert Algorithm 1
   recovers it).
2. Smoke run on a small sampled subset.
3. Full run on real MovieLens-1M, captured to `output/`.
4. Live browser verification (Playwright), zero console errors, before calling the UI done.
