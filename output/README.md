# Output

Evidence from a real, end-to-end run of the application — started with `scripts/app.sh start`,
trained on the full MovieLens 1M dataset, and walked through in the browser with Playwright
(zero console errors, see `screenshots/console_errors.json`). Screenshots reflect the "Network
Lab" UI: a light-first indigo/cyan theme, monospace display type, graph-paper dot-grid texture,
hexagonal node badges, and a top underline-tab nav — the third distinct visual identity in this
workspace (deliberately different from both sibling projects' layouts and palettes).

## A real bug, caught and fixed mid-project

The first implementation of CGNF's community detection (a literal reading of the paper's
Eq. 15–17 as an alternating power-iteration update) silently collapsed to a single cluster
regardless of the target community count — it ran without error and produced a plausible-looking
modularity score. It was caught only because this project's own multi-seed statistical-validation
feature (built to satisfy a reviewer request) produced *bit-identical* results across 5 different
random seeds, which should be essentially impossible for genuine stochastic clustering. Replaced
with bipartite spectral co-clustering (Dhillon, 2001) — verified to produce real, varied,
seed-sensitive community partitions. Full account in `specs/spec.md` §2 and
`review-comments.docx`. Every number below is from the fixed version.

## `results/`

- `run_5ab5fa5a70.json` — full state of the real training run: config, dataset stats (6,040
  users, 3,883 movies, 1,000,209 ratings, 3,563 held-out evaluation users), BPR-MF/LightGCN
  per-epoch loss curves, CGNF's H-sweep modularity curve, the 5-way comparison, CGNF's 4-way
  internal ablation, and diversity/efficiency metrics for every method.
- `multiseed_5ab5fa5a70.json` — 5-seed statistical validation: CGNF vs. Item-kNN (the strongest
  baseline), mean ± SD, and a paired significance test.
- `runs_summary.json` — the `/api/runs` listing at the time of capture.

**Headline result** (NDCG@20, higher is better): CGNF (0.0707) beats every baseline — MostPopular
+115.9%, BPR-MF +74.1%, LightGCN +38.6%, and the strongest baseline, Item-kNN, **+12.6%** —
confirmed statistically significant across 5 seeds (paired t-test, t=33.28, p = 4.9×10⁻⁶).
CGNF's own ablation shows weighted modularity is the component that matters (+4.6% over
unweighted); edge reduction does not help accuracy at this scale (−0.2%) while removing 0.7%
of edges. CGNF does **not** win on every axis: BPR-MF has far higher catalog coverage (69.7%
vs. CGNF's 24.5%) — accuracy and coverage trade off here, reported as measured.

## `screenshots/`

Step-by-step walkthrough, in order:

1. `01_datasets_page` — Datasets page, default MovieLens 1M card plus the custom-upload form.
2. `02_train_page` — Train page after picking a dataset, full hyperparameter form for all 5
   methods (CGNF, Item-kNN, BPR-MF, LightGCN) pre-filled with backend defaults.
3. `03_train_page_dark` — same page in dark mode.
4. `04_results_overview` — run detail header, dataset stats, BPR-MF/LightGCN loss curves, and
   CGNF's modularity-vs-H sweep curve.
5. `05_cgnf_ablation` — the 5-way method comparison bar chart (NDCG@20) and CGNF's 4-way
   internal ablation chart, both with the real run's numbers.
6. `06_recall_metric` — the same comparison switched to the Recall@20 metric tab.
7. `07_diversity_efficiency` — the diversity table (coverage/novelty/ILD per method) and the
   edge-reduction efficiency panel (edge count and wall-clock before/after).
8. `08_recommend_results` — a real top-10 CGNF recommendation for user 1.
9. `09_sensitivity_page` — the interactive `(H, τ)` sensitivity sweep, before computing a cell.
10. `10_sensitivity_computed` — after computing one cell (H=16, τ=0.70), showing real modularity
    and metric values for that specific configuration.
11. `11_validation_page` — the real 5-seed statistical validation result: CGNF 0.0708 ± 0.0005
    vs. Item-kNN 0.0628 ± 0.0000, paired t-test t=33.28, p≈0.

`console_errors.json` is the full list of browser console errors captured during this
walkthrough — empty, confirming a clean run.
