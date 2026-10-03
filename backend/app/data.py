"""Dataset loading for the MovieLens 1M format (movies.dat / ratings.dat,
'::'-delimited), adapted from the sibling hybrid-movie-recommender project's
proven leave-k-out, multi-user evaluation split — see specs/spec.md §2 for
why that honesty bar (held-out split, per-user averaged metrics) matters
here too, independent of it also being the right thing to do.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Dataset:
    name: str
    num_users: int
    num_items: int
    user_org_ids: List
    item_org_ids: List
    item_titles: List[str]
    item_genres: List[List[str]]
    train_user_items: Dict[int, List[int]]
    train_ratings: Dict[int, Dict[int, float]]   # user_idx -> {item_idx: rating}
    train_matrix: sp.csr_matrix                  # (num_users, num_items), raw ratings
    test_user_items: Dict[int, List[int]]         # held out, rating >= threshold
    relevance_threshold: float
    holdout_k: int
    min_interactions: int
    stats: dict = field(default_factory=dict)


def _read_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                      names=["movie_id", "title", "genres"])
    df["genres"] = df["genres"].fillna("").apply(lambda g: [x for x in g.split("|") if x])
    return df


def _read_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="::", engine="python",
                        names=["user_id", "movie_id", "rating", "timestamp"])


def _split(ratings: pd.DataFrame, k: int, min_interactions: int, seed: int):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for _uid, grp in ratings.groupby("user_id", sort=False):
        grp = grp.sort_values("timestamp", kind="stable")
        if len(grp) <= min_interactions:
            train_rows.append(grp)
            continue
        train_rows.append(grp.iloc[:-k])
        test_rows.append(grp.iloc[-k:])
    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else ratings.iloc[0:0]
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else ratings.iloc[0:0]
    _ = rng  # reserved for any future stochastic split variant
    return train_df, test_df


def build(name: str, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, *,
          holdout_k: int = 1, min_interactions: int = 5,
          relevance_threshold: float = 4.0, seed: int = 0) -> Dataset:
    user_ids = sorted(ratings_df["user_id"].unique())
    item_ids = sorted(movies_df["movie_id"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {m: i for i, m in enumerate(item_ids)}

    item_titles = [""] * len(item_ids)
    item_genres: List[List[str]] = [[] for _ in item_ids]
    for _, row in movies_df.iterrows():
        idx = item_index.get(row["movie_id"])
        if idx is not None:
            item_titles[idx] = row["title"]
            item_genres[idx] = row["genres"]

    ratings_df = ratings_df[ratings_df["movie_id"].isin(item_index)].copy()
    train_df, test_df = _split(ratings_df, holdout_k, min_interactions, seed)

    num_users, num_items = len(user_ids), len(item_ids)
    train_user_items: Dict[int, List[int]] = {}
    train_ratings: Dict[int, Dict[int, float]] = {}
    rows, cols, vals = [], [], []
    for r in train_df.itertuples(index=False):
        u, i, rt = user_index[r.user_id], item_index[r.movie_id], float(r.rating)
        train_user_items.setdefault(u, []).append(i)
        train_ratings.setdefault(u, {})[i] = rt
        rows.append(u); cols.append(i); vals.append(rt)
    train_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))

    test_user_items: Dict[int, List[int]] = {}
    for r in test_df.itertuples(index=False):
        if float(r.rating) >= relevance_threshold:
            u, i = user_index[r.user_id], item_index[r.movie_id]
            test_user_items.setdefault(u, []).append(i)

    nnz = train_matrix.nnz
    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_ratings": int(len(ratings_df)),
        "num_train_interactions": int(nnz),
        "num_test_users": len(test_user_items),
        "sparsity": 1.0 - nnz / (num_users * num_items),
    }

    return Dataset(
        name=name, num_users=num_users, num_items=num_items,
        user_org_ids=user_ids, item_org_ids=item_ids,
        item_titles=item_titles, item_genres=item_genres,
        train_user_items=train_user_items, train_ratings=train_ratings,
        train_matrix=train_matrix, test_user_items=test_user_items,
        relevance_threshold=relevance_threshold, holdout_k=holdout_k,
        min_interactions=min_interactions, stats=stats,
    )


def parse_custom(movies_bytes: bytes, ratings_bytes: bytes):
    import io
    movies_df = pd.read_csv(io.BytesIO(movies_bytes))
    ratings_df = pd.read_csv(io.BytesIO(ratings_bytes))
    cols = {c.lower(): c for c in movies_df.columns}
    movies_df = movies_df.rename(columns={
        cols.get("movie_id", "movie_id"): "movie_id",
        cols.get("title", "title"): "title",
        cols.get("genres", "genres"): "genres",
    })
    movies_df["genres"] = movies_df["genres"].fillna("").apply(
        lambda g: [x for x in str(g).split("|") if x])

    rcols = {c.lower(): c for c in ratings_df.columns}
    ratings_df = ratings_df.rename(columns={
        rcols.get("user_id", "user_id"): "user_id",
        rcols.get("movie_id", "movie_id"): "movie_id",
        rcols.get("rating", "rating"): "rating",
    })
    if "rating" not in ratings_df.columns:
        ratings_df["rating"] = 5.0
    if "timestamp" not in ratings_df.columns:
        ratings_df["timestamp"] = np.arange(len(ratings_df))
    return movies_df, ratings_df


def load_movielens(data_dir: Optional[str] = None, dataset_dir: str = "movielens-1m", *,
                    holdout_k: int = 1, min_interactions: int = 5,
                    relevance_threshold: float = 4.0, seed: int = 0,
                    use_cache: bool = True) -> Dataset:
    base = Path(data_dir) if data_dir else (PROJECT_ROOT / "input")
    folder = base / dataset_dir
    cache_dir = (PROJECT_ROOT / "input" / ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{dataset_dir}|{holdout_k}|{min_interactions}|{relevance_threshold}|{seed}"
    cache_file = cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".pkl")
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    movies_df = _read_movies(folder / "movies.dat")
    ratings_df = _read_ratings(folder / "ratings.dat")
    ds = build(dataset_dir, movies_df, ratings_df, holdout_k=holdout_k,
               min_interactions=min_interactions, relevance_threshold=relevance_threshold,
               seed=seed)
    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(ds, f)
    return ds
