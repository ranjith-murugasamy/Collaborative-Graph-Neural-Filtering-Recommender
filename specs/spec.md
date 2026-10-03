# CGNF Recommender — Specification

**Status:** Draft v1 · **Owner:** generated from source material below · **Last updated:** 2026-10-04

## 0. Source material

The following requirements were derived from the source material below; §2 and §6 cite the
specific claims and gaps this spec responds to.

1. **`Enhancing Recommendation Systems_shruthik_INASS_Analysis_updated.docx.pdf`** — the paper
   *"Enhancing Recommendation Systems through Novel Collaborative Graph Neural Filtering
   Approach"* (CGNF), Shruthi K C & Geeta C Mara, REVA University — plus two rounds of reviewer
   comments (12 items from Reviewer 1, 8 from Reviewer 2), a "Work To Be Done" revision plan,
   and a revision-cost quote.
2. **`mymedialite/`** — [MyMediaLite](http://mymedialite.net), an established open-source
   recommender-algorithm library (Gantner et al., RecSys 2011). Used here as a methodological
   reference for classic baseline definitions (MostPopular, Item/User-kNN, BPR-MF) — its
   canonical hyperparameter conventions (e.g. BPR-MF's default learning rate 0.05) ground this
   project's reimplementations, not its C#/.NET source directly (see §6).

Unlike the sibling `hybrid-movie-recommender` project, **no runnable reference implementation
of CGNF itself exists** — only the paper's prose, equations (1)–(17), and three pseudocode
algorithms. Faithful reproduction here means transcribing the paper's own mathematics, not
auditing someone else's code.

## 1. Purpose

This application exists to close the gap between the paper's claims and what can actually be
verified: build CGNF — a weighted, dual-mode bipartite community-detection pipeline feeding a
neighborhood-based collaborative filtering step — directly from its equations, compare it
honestly against classic and recent baselines on real data with a leakage-free protocol, and
make every number the paper currently reports inconsistently (see §2) reproducible from one
consistent, inspectable source instead.

## 2. What the source material actually provides (and why reproduction starts from zero)

Two kinds of gaps exist in the reference material: what the reviewers asked to fix, and what
close reading of the paper itself revealed independent of either reviewer. This section covers
the latter.

1. **No reference code.** The paper describes CGNF entirely in mathematical and pseudocode form
   (Algorithms 1–3, equations 1–17). There is nothing to run, so there is no "two bugs found in
   the reference implementation" story like the sibling project's — the risk here is pure
   transcription fidelity, which §4.2 addresses by implementing every numbered equation as a
   named, independently testable function.
2. **The paper's own self-reported numbers are internally inconsistent in at least five
   places** — found independently of, but matching, Reviewer 1's items 5–9:
   - Amazon Recall@50 for CGNF: **22.67** in Table 2 vs. **21.84** in the surrounding prose.
   - MovieLens Recall@50 for CGNF: **47.05** in Figure 6's caption text, **49.85** and **49.86**
     in two other prose sentences, and **49.86** again in Table 7 — four different values for
     one claimed result.
   - Table 5/6 and Figure 7/8 are captioned "NDCG for MovieLens," while the surrounding prose
     narrates "the NDCG@20 is evaluated for Amazon Dataset" — the dataset label contradicts
     itself within the same subsection.
   - Table 7's "Improvisation" column (e.g. "74.3367%") does not reproduce from
     `(CGNF − baseline) / baseline × 100` applied to the adjacent columns' own numbers.
   - MovieLens Recall@20's existing-system baseline is given as 10.985 in Table 3 and 10.895 in
     the Table 7 discussion — a sixth, smaller inconsistency on top of the five above.

Given this, picking "the correct" number out of the paper is not a meaningful fix — there is no
way to know which of 22.67/21.84 or 47.05/49.85/49.86 the authors actually measured. **This
application does not attempt to arbitrate the paper's numbers.** It reproduces CGNF and its
baselines from scratch, on a documented, leakage-free protocol, and reports whatever that
reproduction actually measures — a single, internally consistent set of numbers traceable to
one run configuration, replacing five contradictory ones.

3. **A third, self-found issue — this time in this project's own first implementation, not the
   paper.** An initial reading of Eq.(15)–(17)/Algorithm 1 as an alternating power-iteration
   update was built, trained on real MovieLens-1M data, and looked plausible — until this
   project's own multi-seed statistical-validation feature (REQ-19, built to answer Reviewer 2
   item 5) produced *bit-identical* results across 5 different random seeds, which should be
   essentially impossible for genuine stochastic clustering. Direct investigation confirmed
   every tested community count H collapsed to exactly one occupied community regardless of H
   — a known failure mode of repeated multiplication by a positive matrix on one connected
   graph (it converges to the dominant Perron eigenvector, not H distinct clusters). Fixed by
   replacing it with bipartite spectral co-clustering (Dhillon, 2001), the standard, correct
   method for this exact objective — see `backend/app/graph.py`'s module docstring for the full
   account. Documented here because the honesty standard this spec sets for the *paper* has to
   apply to this *implementation* too, including when the implementation finds its own bug.

## 3. Users and use cases

- **The researcher (primary user)**: loads a dataset, runs CGNF and the baselines under
  identical conditions, explores CGNF's own internal ablations and sensitivity to its two main
  hyperparameters (community count, edge-reduction threshold), and reads off a results table
  that — for the first time — agrees with itself.
- **A reviewer or collaborator**: opens the dashboard to check a specific number's provenance
  (which run, which seed, which protocol) without re-deriving it from five conflicting
  manuscript passages.

## 4. Functional requirements

### 4.1 Data
- REQ-1: Ship a default, real dataset in `input/` — MovieLens 1M (6,040 users, 3,706 movies,
  1,000,209 ratings), the same format and scale the paper itself evaluates on.
- REQ-2: Build the user–item bipartite interaction matrix and report its sparsity/degree
  statistics before any training run, per Reviewer 2's "dataset characteristics" ask.
- REQ-3: Accept a custom ratings/movies upload, parsed into the same internal representation.

### 4.2 Graph construction & community detection (CGNF core)
- REQ-4: Construct the dual-mode (user/item) bipartite graph and its incidence matrix *D*, per
  Eq. (3).
- REQ-5: Compute the paper's custom weighted dual-mode modularity (Eq. 4–14) — not standard
  Louvain modularity (see §6) — including the degree-normalized prediction terms
  `pred_L(U_v,l)`/`pred_V(U_v,l)` (Eq. 10–11) and the local/global modularity blend `Z^Gl_x`
  (Eq. 14).
- REQ-6: Iterative community assignment (Algorithm 1) with configurable `H_min`≈2 and
  `H_max = (|V|·|L|)^0.5`, per the paper's own stated bounds.
- REQ-7: Edge reduction (Algorithm 2) removing redundant within-community links, with the
  before/after edge count retained for REQ-18 (efficiency reporting).
- REQ-8: Neighborhood-based score prediction (Algorithm 3) — find the target user's community,
  rank unrated items by neighbor-weighted similarity.
- REQ-9: Every sub-step in REQ-5–REQ-8 individually toggleable (weighted vs. unweighted
  modularity; edge reduction on/off) from one shared pipeline function — not separate
  "ablation-mode" code paths — so REQ-20's ablation reuses exactly the code REQ-4–8 run by
  default.

### 4.3 Baseline recommenders
- REQ-10: **MostPopular** — global popularity ranking (matches the paper's own weakest
  baseline and MyMediaLite's `MostPopular.cs` definition).
- REQ-11: **Item-kNN** — cosine-similarity collaborative item neighborhoods, MyMediaLite's
  `ItemKNN.cs` definition.
- REQ-12: **BPR-MF** — pairwise ranking matrix factorization (Rendle et al. 2009), matching
  MyMediaLite's `BPRMF.cs` default learning rate (0.05) and loss family.
- REQ-13: **LightGCN** — the specific "modern collaborative approach" both reviewers name
  explicitly (He et al. 2020): embeddings propagated through *K* layers of the normalized
  bipartite adjacency, final embedding = the unweighted mean of all layers' embeddings, trained
  with the same BPR loss as REQ-12. Implemented, not deferred — see §6 for why this is NumPy,
  not PyTorch.

### 4.4 Comparative evaluation & ablation
- REQ-14: Recall@K and NDCG@K at **K ∈ {20, 50}** (matching the paper's own K values, not the
  sibling project's {5,10,15,20}) computed per held-out user and averaged over every held-out
  user with ≥1 relevant item — not a global/pooled average (the paper's own §4.1 states it
  prefers "a more interpretable average per user," which this implementation actually does).
- REQ-15: One-click, five-way comparison — MostPopular, Item-kNN, BPR-MF, LightGCN, CGNF — all
  under the identical split, candidate set, and metric implementation.
- REQ-16: CGNF's own internal ablation (REQ-9's toggles): full CGNF vs. no edge reduction vs.
  unweighted modularity vs. both disabled, answering "which of CGNF's own pieces matter" —
  Reviewer 1 item 11 / Reviewer 2 item 3.

### 4.5 Diversity evaluation
- REQ-17: Coverage (fraction of the catalog appearing in any user's Top-K), novelty
  (popularity-based self-information of recommended items), and intra-list diversity (mean
  pairwise genre dissimilarity within a user's own Top-K) — computed for every method in
  REQ-15, not claimed from Recall/NDCG alone (Reviewer 1 item 4 / Reviewer 2 item 4 explicitly
  say accuracy metrics cannot support a diversity claim).

### 4.6 Efficiency evaluation
- REQ-18: Report edge count and wall-clock time before and after edge reduction (REQ-7), not
  just assert the reduction "boosts computational efficiency" as the abstract currently does.

### 4.7 Statistical validation
- REQ-19: Multi-seed runs (configurable count, default 5) with mean ± standard deviation per
  metric, plus a paired significance test (Wilcoxon signed-rank, since per-user metrics are not
  normally distributed) between CGNF and the strongest baseline from REQ-15.

### 4.8 Sensitivity analysis
- REQ-20: Interactive sweep over CGNF's two main hyperparameters — target community count and
  the edge-reduction threshold `τ` — re-evaluated from cached per-user neighbor sets where
  possible, full re-run otherwise (community structure changes propagate further than a fusion
  weight does, so this cannot always be instant the way the sibling project's weight sliders
  are — see `plan.md`).

### 4.9 Application
- REQ-21: Web UI covering dataset selection/upload, run configuration, live progress, a
  five-way comparison dashboard, CGNF's own ablation panel, diversity/efficiency panels, and
  the sensitivity sweep.
- REQ-22: REST + streaming API backing the UI, usable headlessly for scripted experiments.
- REQ-23: **Single script to set up, start, and stop the whole application.**

## 5. Non-functional requirements

- NFR-1: Runs on stock local Python — no GPU, no deep-learning framework dependency — see §6
  for what was checked and why NumPy/scipy was chosen anyway, not because anything was broken.
- NFR-2: A full run (all 5 methods + ablation + default sensitivity grid, default dataset)
  completes in minutes on CPU, not hours — verified against a real run, not estimated (see
  `output/`).
- NFR-3: Every number shown in the UI is traceable to a specific run configuration — directly
  answering §2's "which of the paper's five conflicting numbers is right" problem by making
  that question unaskable: there is exactly one number per metric per run.

## 6. Why NumPy/scipy instead of networkx, python-louvain, or PyTorch

Checked empirically against this environment before writing any code — all three **install
cleanly** here (`pip install --dry-run networkx python-louvain torch` succeeds, resolving
`networkx-3.7`, `python-louvain-0.16`, `torch-2.14.1`). This is a different situation from the
sibling project, where LightFM and stable TensorFlow genuinely do not install — here, the
decision not to use them is architectural, not forced:

- **`networkx` / `python-louvain` implement standard single-mode Newman–Girvan modularity.**
  CGNF's modularity (Eq. 4–14) is a custom **weighted, dual-mode** (bipartite) formulation with
  edge-reduction folded into the same objective — a generic community-detection library would
  not produce the algorithm the paper describes, only something that looks superficially
  similar. Faithfully reproducing (and being able to critique) CGNF specifically requires
  implementing its actual equations, not a library's version of a related-but-different idea.
- **PyTorch would work for LightGCN**, but every other component in this application (and the
  sibling project's) is hand-implemented NumPy with gradient-checked backward passes. Pulling
  in a full deep-learning framework for exactly one of five methods would make that one method
  less inspectable than the other four, not more — and LightGCN's actual computation (repeated
  sparse-matrix propagation plus a BPR loss) has no real need for autodiff. Implemented in
  NumPy instead, validated the same way as everything else: gradient-checked before being
  trusted on real data.

## 7. Items explicitly out of scope

Explicitly deferred, with reasons, rather than silently dropped from the implementation:

- **The Amazon (book-review) dataset.** The paper evaluates on both MovieLens-1M and an Amazon
  book-review subset. Amazon review data requires separate licensing/acquisition and materially
  different preprocessing (review text, variable-length sequences) than MovieLens — a
  second, comparably-sized workstream. This application ships MovieLens-1M only; the dataset
  abstraction (REQ-1/REQ-3) accepts a second dataset later without redesign.
- **The session-based / multi-interest baselines** (YouTube DNN, GRU4Rec, MIND, ComiRec-DR,
  ComiRec-SA, FAT). These require sequential session data and substantially different
  architectures (RNNs, capsule networks, dynamic routing) from the static user–item matrix
  CGNF and the four implemented baselines (§4.3) operate on — a separate project, not a
  same-shape extension. LightGCN was implemented instead as the one explicitly-named "recent
  graph-based method" both reviewers asked for (Reviewer 1 item 3, Reviewer 2 item 1/8).
- **Manuscript-only asks**: the proposed Section 1–5 restructuring, the 2024–2026 literature
  review and its comparison table, and reference-list completion. These are writing tasks with
  no corresponding application feature — out of scope for an application spec; they belong in
  the manuscript, not the codebase.

## 8. Acceptance criteria

| Criterion | How verified |
|---|---|
| A fresh checkout runs end-to-end via the single setup/start script with no manual steps. | `scripts/app.sh start`, observed clean. |
| All 5 methods (REQ-15) produce Recall@{20,50} and NDCG@{20,50} on one consistent, held-out, per-user-averaged protocol. | Real run exported to `output/results/`. |
| CGNF's internal ablation (REQ-16) and the sensitivity sweep (REQ-20) both run from the UI without hand-editing config files. | Screenshots in `output/screenshots/`. |
| Diversity (REQ-17) and efficiency (REQ-18) numbers exist for every method, not asserted in prose. | Same run export. |
| Multi-seed statistical validation (REQ-19) reports mean ± SD and a significance test, not a single point estimate. | Same run export. |
| Every one of the paper's 5–6 internal numeric inconsistencies (§2) is replaced by exactly one number per metric, from one run. | `review-comments.docx`, Appendix A. |
