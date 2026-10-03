"""Flask REST + SSE API — same shape as the sibling hybrid-movie-recommender
project's api.py (health/config/datasets/runs/logs/stream), plus CGNF-
specific routes: sensitivity sweep cells and multi-seed statistical
validation.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from . import cgnf as cgnf_mod
from . import custom_store
from . import data as data_mod
from .runner import CGNF_ABLATION_PRESETS, DEFAULTS, METHOD_LABELS, manager

app = Flask(__name__)
CORS(app)

_default_dataset_cache = None


def _default_dataset_summary():
    global _default_dataset_cache
    if _default_dataset_cache is None:
        ds = data_mod.load_movielens()
        _default_dataset_cache = {
            "id": "movielens-1m", "label": "MovieLens 1M", "source": "default",
            "num_users": ds.num_users, "num_items": ds.num_items,
            "num_ratings": ds.stats["num_ratings"],
        }
    return _default_dataset_cache


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/config/defaults")
def config_defaults():
    return jsonify({
        **DEFAULTS,
        "method_labels": METHOD_LABELS,
        "cgnf_ablation_presets": {k: {"label": v["label"],
                                      "weighted_modularity": v["weighted_modularity"],
                                      "use_edge_reduction": v["use_edge_reduction"]}
                                   for k, v in CGNF_ABLATION_PRESETS.items()},
    })


@app.get("/api/datasets")
def list_datasets():
    return jsonify([_default_dataset_summary(), *custom_store.list_all()])


@app.post("/api/datasets/custom")
def upload_dataset():
    movies_file = request.files.get("movies")
    ratings_file = request.files.get("ratings")
    if not movies_file or not ratings_file:
        return jsonify({"error": "movies and ratings files are required"}), 400
    name = request.form.get("name", "")
    meta = custom_store.create(name, movies_file.read(), ratings_file.read())
    return jsonify(meta)


_INT_FIELDS = ["holdout_k", "min_interactions", "seed", "cgnf_iters", "cgnf_max_h",
               "cgnf_top_k_neighbors", "itemknn_k", "bprmf_embedding_dim", "bprmf_epochs",
               "bprmf_batch_size", "lightgcn_embedding_dim", "lightgcn_num_layers",
               "lightgcn_epochs", "lightgcn_batch_size", "diversity_top_k"]
_FLOAT_FIELDS = ["relevance_threshold", "cgnf_tau", "bprmf_lr", "bprmf_lr_decay",
                  "bprmf_l2_coef", "lightgcn_lr", "lightgcn_lr_decay", "lightgcn_l2_coef"]


@app.post("/api/runs")
def create_run():
    body = request.get_json(force=True)
    cfg = dict(DEFAULTS)
    cfg.update(body)
    cfg.setdefault("dataset", {"source": "default", "id": "movielens-1m"})
    for f in _INT_FIELDS:
        if f in cfg and cfg[f] is not None:
            cfg[f] = int(cfg[f])
    for f in _FLOAT_FIELDS:
        if f in cfg and cfg[f] is not None:
            cfg[f] = float(cfg[f])
    run = manager.create(cfg)
    return jsonify({"run_id": run.id})


@app.get("/api/runs")
def list_runs():
    return jsonify(manager.list_summaries())


@app.get("/api/runs/<run_id>")
def get_run(run_id):
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(run.to_public_dict())


@app.get("/api/runs/<run_id>/logs")
def get_logs(run_id):
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    after = int(request.args.get("after", 0))
    return jsonify({"logs": run.logs_after(after)})


@app.get("/api/runs/<run_id>/stream")
def stream_run(run_id):
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404

    def gen():
        seq = 0
        while True:
            run.updated.wait(timeout=1.0)
            run.updated.clear()
            new_logs = run.logs_after(seq)
            if new_logs:
                seq = new_logs[-1]["seq"] + 1
                yield f"event: logs\ndata: {json.dumps(new_logs)}\n\n"
            yield f"event: status\ndata: {json.dumps(run.to_public_dict())}\n\n"
            if run.status in ("finished", "error") and not run.logs_after(seq):
                break

    return Response(gen(), mimetype="text/event-stream")


@app.get("/api/runs/<run_id>/sensitivity")
def sensitivity(run_id):
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    if run.status != "finished":
        return jsonify({"error": "run not finished"}), 409
    H = int(request.args.get("H", 16))
    tau = float(request.args.get("tau", 0.7))
    cell = manager.sensitivity_cell(run, H, tau)
    return jsonify(cell)


@app.post("/api/runs/<run_id>/multiseed")
def multiseed(run_id):
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    body = request.get_json(silent=True) or {}
    num_seeds = int(body.get("seeds", 5))
    metric_key = body.get("metric", "ndcg@20")

    import threading
    def go():
        try:
            manager.run_multiseed(run, num_seeds=num_seeds, metric_key=metric_key)
        except Exception as e:
            run.log(f"ERROR (multiseed): {type(e).__name__}: {e}")

    threading.Thread(target=go, daemon=True).start()
    return jsonify({"status": "started"})


@app.get("/api/runs/<run_id>/movies/search")
def search_movies(run_id):
    run = manager.get(run_id)
    if run is None or run._dataset is None:
        return jsonify({"error": "not found"}), 404
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    ds = run._dataset
    results = []
    for idx, title in enumerate(ds.item_titles):
        if q in title.lower():
            results.append({"item_index": idx, "title": title, "genres": ds.item_genres[idx]})
            if len(results) >= 20:
                break
    return jsonify(results)


@app.get("/api/runs/<run_id>/recommend")
def recommend(run_id):
    run = manager.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    if run.status != "finished":
        return jsonify({"error": "run not finished"}), 409
    user_id = int(request.args.get("user_id", 1))
    top_k = int(request.args.get("top_k", 10))
    method = request.args.get("method", "cgnf")

    ds = run._dataset
    try:
        user_idx = ds.user_org_ids.index(user_id)
    except ValueError:
        return jsonify({"error": f"unknown user_id {user_id}"}), 400

    if method == "cgnf":
        scores = cgnf_mod.score_for_user(run._models["cgnf"], user_idx,
                                          run.config["cgnf_top_k_neighbors"])
    elif method in run._models:
        scores = run._models[method].score_for_user(user_idx)
    else:
        return jsonify({"error": f"unknown method {method}"}), 400

    scores = scores.copy()
    seen = ds.train_user_items.get(user_idx, [])
    if seen:
        scores[seen] = -np.inf
    top_idx = np.argpartition(-scores, kth=min(top_k, len(scores) - 1) - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    items = [{
        "item_index": int(i), "title": ds.item_titles[i], "genres": ds.item_genres[i],
        "score": float(scores[i]),
    } for i in top_idx]
    return jsonify({"user_id": user_id, "method": method, "items": items})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("CGNF_PORT", 5003)),
             debug=True, use_reloader=False, threaded=True)
