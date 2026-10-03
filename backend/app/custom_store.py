"""Uploaded-dataset storage — same pattern as the sibling project's
custom_store.py."""
from __future__ import annotations

import json
import pickle
import time
import uuid
from pathlib import Path

from . import data as data_mod

STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage" / "datasets"


def create(display_name: str, movies_bytes: bytes, ratings_bytes: bytes) -> dict:
    movies_df, ratings_df = data_mod.parse_custom(movies_bytes, ratings_bytes)
    dataset_id = uuid.uuid4().hex[:10]
    folder = STORAGE_DIR / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "frames.pkl", "wb") as f:
        pickle.dump({"movies": movies_df, "ratings": ratings_df}, f)
    meta = {
        "id": dataset_id, "label": display_name or f"Custom dataset {dataset_id}",
        "source": "custom", "created_at": time.time(),
        "num_users": int(ratings_df["user_id"].nunique()),
        "num_items": int(movies_df["movie_id"].nunique()),
        "num_ratings": int(len(ratings_df)),
    }
    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f)
    return meta


def list_all() -> list:
    if not STORAGE_DIR.exists():
        return []
    out = []
    for folder in sorted(STORAGE_DIR.iterdir()):
        meta_file = folder / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                out.append(json.load(f))
    return out


def load(dataset_id: str, **split_kwargs):
    folder = STORAGE_DIR / dataset_id
    with open(folder / "frames.pkl", "rb") as f:
        frames = pickle.load(f)
    with open(folder / "metadata.json") as f:
        meta = json.load(f)
    return data_mod.build(meta["label"], frames["movies"], frames["ratings"], **split_kwargs)
