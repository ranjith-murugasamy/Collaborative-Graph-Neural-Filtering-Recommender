"""Smoke test: a small sampled subset of real MovieLens-1M, all 5 methods,
before trusting a full-scale run."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import cgnf as cgnf_mod
from app import data as data_mod
from app import diversity as diversity_mod
from app import graph as graph_mod
from app import metrics as metrics_mod
from app.baselines import bprmf, itemknn, lightgcn, mostpop


def main():
    ds_full = data_mod.load_movielens(use_cache=False)
    rng = np.random.default_rng(0)
    sample_users = rng.choice(ds_full.num_users, size=400, replace=False)
    # Build a small sub-dataset by filtering ratings to sampled users, reusing build().
    import pandas as pd
    movies_df = pd.DataFrame({
        "movie_id": ds_full.item_org_ids,
        "title": ds_full.item_titles,
        "genres": ds_full.item_genres,
    })
    rows = []
    user_set = set(ds_full.user_org_ids[i] for i in sample_users)
    # Re-read raw ratings for just these users (cheap enough at this scale).
    raw = data_mod._read_ratings(data_mod.PROJECT_ROOT / "input" / "movielens-1m" / "ratings.dat")
    raw = raw[raw["user_id"].isin(user_set)]
    ds = data_mod.build("smoke", movies_df, raw, holdout_k=1, min_interactions=5,
                         relevance_threshold=4.0, seed=0)
    print(f"smoke dataset: users={ds.num_users} items={ds.num_items} "
          f"train={ds.stats['num_train_interactions']} eval_users={ds.stats['num_test_users']}")

    t0 = time.perf_counter()
    mp = mostpop.fit(ds.train_matrix)
    iknn = itemknn.fit(ds.train_matrix, k=20)
    bpr = bprmf.fit(ds.train_matrix, embedding_dim=16, epochs=5, batch_size=512, seed=0)
    lgcn = lightgcn.fit(ds.train_matrix, embedding_dim=16, num_layers=2, epochs=5, batch_size=512, seed=0)
    cgnf_model = cgnf_mod.fit(ds.train_matrix, h_grid=graph_mod.practical_h_grid(ds.num_users, ds.num_items, 16),
                               iters=8, seed=0)
    print(f"training all 5 methods took {time.perf_counter()-t0:.1f}s, CGNF H*={cgnf_model.H}")

    fns = {
        "mostpop": lambda u: mp.score_for_user(u),
        "itemknn": lambda u: iknn.score_for_user(u),
        "bprmf": lambda u: bpr.score_for_user(u),
        "lightgcn": lambda u: lgcn.score_for_user(u),
        "cgnf": lambda u: cgnf_mod.score_for_user(cgnf_model, u, 20),
    }
    item_counts = np.asarray((ds.train_matrix > 0).sum(axis=0)).ravel().astype(np.float64)
    for name, fn in fns.items():
        m = metrics_mod.evaluate(fn, ds.train_user_items, ds.test_user_items, ds.num_items, k_list=[20, 50])
        dv = diversity_mod.evaluate_diversity(fn, ds.train_user_items, ds.test_user_items, ds.item_genres,
                                               item_counts, top_k=20)
        print(f"[{name}] recall@20={m['recall@20']:.4f} ndcg@20={m['ndcg@20']:.4f} "
              f"recall@50={m['recall@50']:.4f} ndcg@50={m['ndcg@50']:.4f} "
              f"coverage={dv['coverage']:.3f} novelty={dv['novelty']:.3f} ild={dv['intra_list_diversity']:.3f}")

    print("smoke test OK")


if __name__ == "__main__":
    main()
