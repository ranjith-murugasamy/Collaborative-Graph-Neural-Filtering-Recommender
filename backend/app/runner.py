"""Run orchestration: trains all 5 methods (MostPopular, Item-kNN, BPR-MF,
LightGCN, CGNF) plus CGNF's internal ablation on one held-out split, then
diversity/efficiency, exposing live state the same way the sibling
hybrid-movie-recommender project's runner.py does (background thread,
lock-guarded state, SSE-friendly log buffer).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import cgnf as cgnf_mod
from . import custom_store
from . import data as data_mod
from . import diversity as diversity_mod
from . import graph as graph_mod
from . import metrics as metrics_mod
from . import stats as stats_mod
from .baselines import bprmf, itemknn, lightgcn, mostpop

METHOD_LABELS = {
    "mostpop": "MostPopular",
    "itemknn": "Item-kNN",
    "bprmf": "BPR-MF",
    "lightgcn": "LightGCN",
    "cgnf": "CGNF (proposed)",
}

CGNF_ABLATION_PRESETS = {
    "full": {"weighted_modularity": True, "use_edge_reduction": True,
             "label": "Full CGNF"},
    "no_edge_reduction": {"weighted_modularity": True, "use_edge_reduction": False,
                           "label": "CGNF − edge reduction"},
    "unweighted_modularity": {"weighted_modularity": False, "use_edge_reduction": True,
                               "label": "CGNF − weighted modularity"},
    "neither": {"weighted_modularity": False, "use_edge_reduction": False,
                "label": "CGNF − both (plain community CF)"},
}

DEFAULTS = dict(
    holdout_k=1, min_interactions=5, relevance_threshold=4.0, seed=0,
    cgnf_tau=0.7, cgnf_iters=12, cgnf_max_h=64, cgnf_top_k_neighbors=50,
    itemknn_k=40,
    bprmf_embedding_dim=32, bprmf_epochs=10, bprmf_batch_size=16384,
    bprmf_lr=0.05, bprmf_lr_decay=0.995, bprmf_l2_coef=1e-5,
    # lightgcn_batch_size is deliberately much larger than bprmf's: every
    # batch re-propagates through the *entire* graph (LightGCN has no
    # notion of a mini-batch subgraph the way node-sampled GNN training
    # does), so batch count directly multiplies wall-clock cost. Measured
    # on real MovieLens-1M: ~27s/epoch at batch=4096 vs ~10s/epoch at
    # batch=16384, for the same convergence trajectory.
    lightgcn_embedding_dim=32, lightgcn_num_layers=3, lightgcn_epochs=10,
    lightgcn_batch_size=16384, lightgcn_lr=0.01, lightgcn_lr_decay=0.995,
    lightgcn_l2_coef=1e-5,
    k_list=[20, 50], diversity_top_k=20,
)


class Run:
    def __init__(self, run_id: str, config: dict):
        self.id = run_id
        self.config = config
        self.status = "queued"
        self.created_at = time.time()
        self.dataset_info: dict = {}
        self.logs: List[dict] = []
        self._log_seq = 0
        self.loss_history: Dict[str, list] = {"bprmf": [], "lightgcn": [], "cgnf_h_sweep": []}
        self.results: Dict[str, dict] = {}           # method -> {label, metrics, diversity}
        self.cgnf_ablation: Dict[str, dict] = {}      # combo -> {label, metrics, edges_before, edges_after, seconds}
        self.sensitivity_cache: Dict[str, dict] = {}  # "H:tau" -> {metrics, ...}
        self.multiseed: Optional[dict] = None
        self.current_stage = "queued"
        self.error: Optional[str] = None
        self.lock = threading.RLock()
        self.updated = threading.Event()

        self._dataset = None
        self._models: Dict[str, object] = {}
        self._liked_by_user: Dict[int, List[int]] = {}

    def log(self, message: str):
        with self.lock:
            entry = {"seq": self._log_seq, "t": time.time(), "message": message}
            self._log_seq += 1
            self.logs.append(entry)
        self.updated.set()

    def logs_after(self, seq: int) -> List[dict]:
        with self.lock:
            return [e for e in self.logs if e["seq"] >= seq]

    def to_public_dict(self, include_logs: bool = False) -> dict:
        with self.lock:
            d = {
                "id": self.id, "config": self.config, "status": self.status,
                "created_at": self.created_at, "dataset_info": self.dataset_info,
                "loss_history": self.loss_history, "results": self.results,
                "cgnf_ablation": self.cgnf_ablation, "multiseed": self.multiseed,
                "current_stage": self.current_stage, "error": self.error,
                "log_count": len(self.logs),
            }
            if include_logs:
                d["logs"] = list(self.logs)
            return d


class RunManager:
    def __init__(self):
        self.runs: Dict[str, Run] = {}
        self._lock = threading.Lock()

    def create(self, config: dict) -> Run:
        run_id = uuid.uuid4().hex[:10]
        run = Run(run_id, config)
        with self._lock:
            self.runs[run_id] = run
        thread = threading.Thread(target=self._train, args=(run,), daemon=True)
        thread.start()
        return run

    def get(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def list_summaries(self) -> List[dict]:
        return [r.to_public_dict() for r in self.runs.values()]

    # -- training pipeline -------------------------------------------------

    def _resolve_dataset(self, cfg: dict):
        ds_cfg = cfg["dataset"]
        if ds_cfg.get("source") == "custom":
            return custom_store.load(
                ds_cfg["id"], holdout_k=cfg["holdout_k"], min_interactions=cfg["min_interactions"],
                relevance_threshold=cfg["relevance_threshold"], seed=cfg["seed"])
        return data_mod.load_movielens(
            dataset_dir=ds_cfg.get("id", "movielens-1m"), holdout_k=cfg["holdout_k"],
            min_interactions=cfg["min_interactions"], relevance_threshold=cfg["relevance_threshold"],
            seed=cfg["seed"])

    def _train(self, run: Run):
        cfg = run.config
        try:
            run.status = "running"
            run.current_stage = "loading_dataset"
            run.log(f"Loading dataset ({cfg['dataset'].get('source', 'default')}:{cfg['dataset'].get('id')})...")
            dataset = self._resolve_dataset(cfg)
            run._dataset = dataset
            run.dataset_info = dict(dataset.stats)
            run.log(f"users={dataset.num_users} items={dataset.num_items} "
                     f"train_interactions={dataset.stats['num_train_interactions']} "
                     f"eval_users={dataset.stats['num_test_users']}")

            train_matrix = dataset.train_matrix
            item_counts = np.asarray((train_matrix > 0).sum(axis=0)).ravel().astype(np.float64)

            # --- MostPopular ---
            run.current_stage = "mostpop"
            run.log("Training MostPopular...")
            run._models["mostpop"] = mostpop.fit(train_matrix)

            # --- Item-kNN ---
            run.current_stage = "itemknn"
            run.log(f"Training Item-kNN (k={cfg['itemknn_k']})...")
            run._models["itemknn"] = itemknn.fit(train_matrix, k=cfg["itemknn_k"])

            # --- BPR-MF ---
            run.current_stage = "bprmf"
            run.log(f"Training BPR-MF ({cfg['bprmf_epochs']} epochs, dim={cfg['bprmf_embedding_dim']})...")

            def on_bprmf_epoch(epoch, loss):
                run.loss_history["bprmf"].append({"epoch": epoch, "loss": loss})
                run.log(f"[bprmf] epoch {epoch}: loss={loss:.4f}")

            run._models["bprmf"] = bprmf.fit(
                train_matrix, embedding_dim=cfg["bprmf_embedding_dim"], epochs=cfg["bprmf_epochs"],
                batch_size=cfg["bprmf_batch_size"], lr=cfg["bprmf_lr"], lr_decay=cfg["bprmf_lr_decay"],
                l2_coef=cfg["bprmf_l2_coef"], seed=cfg["seed"], on_epoch=on_bprmf_epoch)

            # --- LightGCN ---
            run.current_stage = "lightgcn"
            run.log(f"Training LightGCN ({cfg['lightgcn_epochs']} epochs, "
                     f"layers={cfg['lightgcn_num_layers']}, dim={cfg['lightgcn_embedding_dim']})...")

            def on_lightgcn_epoch(epoch, loss):
                run.loss_history["lightgcn"].append({"epoch": epoch, "loss": loss})
                run.log(f"[lightgcn] epoch {epoch}: loss={loss:.4f}")

            run._models["lightgcn"] = lightgcn.fit(
                train_matrix, embedding_dim=cfg["lightgcn_embedding_dim"],
                num_layers=cfg["lightgcn_num_layers"], epochs=cfg["lightgcn_epochs"],
                batch_size=cfg["lightgcn_batch_size"], lr=cfg["lightgcn_lr"],
                lr_decay=cfg["lightgcn_lr_decay"], l2_coef=cfg["lightgcn_l2_coef"],
                seed=cfg["seed"], on_epoch=on_lightgcn_epoch)

            # --- CGNF ablation (4 combos; "full" is the headline CGNF model) ---
            run.current_stage = "cgnf"
            for combo_key, combo_cfg in CGNF_ABLATION_PRESETS.items():
                run.log(f"Training CGNF [{combo_cfg['label']}]...")
                t0 = time.perf_counter()

                def on_h_step(H, modularity, _combo=combo_key):
                    run.log(f"[cgnf:{_combo}] H={H} modularity={modularity:.4f}")
                    if _combo == "full":
                        run.loss_history["cgnf_h_sweep"].append({"H": int(H), "modularity": float(modularity)})

                model = cgnf_mod.fit(
                    train_matrix, tau=cfg["cgnf_tau"],
                    h_grid=graph_mod.practical_h_grid(dataset.num_users, dataset.num_items, cfg["cgnf_max_h"]),
                    weighted_modularity=combo_cfg["weighted_modularity"],
                    use_edge_reduction=combo_cfg["use_edge_reduction"],
                    iters=cfg["cgnf_iters"], seed=cfg["seed"], on_h_step=on_h_step)
                elapsed = time.perf_counter() - t0

                if combo_key == "full":
                    run._models["cgnf"] = model

                score_fn = lambda u, _m=model: cgnf_mod.score_for_user(_m, u, cfg["cgnf_top_k_neighbors"])
                m = metrics_mod.evaluate(score_fn, dataset.train_user_items, dataset.test_user_items,
                                          dataset.num_items, k_list=cfg["k_list"])
                run.cgnf_ablation[combo_key] = {
                    "label": combo_cfg["label"], "metrics": m, "H": model.H,
                    "modularity": model.modularity, "edges_before": model.edges_before,
                    "edges_after": model.edges_after, "seconds": elapsed,
                }
                run.log(f"[cgnf:{combo_key}] H*={model.H} edges {model.edges_before}->{model.edges_after} "
                         f"({elapsed:.1f}s) ndcg@{cfg['k_list'][0]}={m.get(f'ndcg@{cfg['k_list'][0]}', 0):.4f}")

            # --- Evaluate all 5 headline methods on the identical protocol ---
            run.current_stage = "evaluating"
            run.log("Evaluating all methods (Recall/NDCG, per-user averaged)...")
            score_fns = {
                "mostpop": lambda u: run._models["mostpop"].score_for_user(u),
                "itemknn": lambda u: run._models["itemknn"].score_for_user(u),
                "bprmf": lambda u: run._models["bprmf"].score_for_user(u),
                "lightgcn": lambda u: run._models["lightgcn"].score_for_user(u),
                "cgnf": lambda u: cgnf_mod.score_for_user(run._models["cgnf"], u, cfg["cgnf_top_k_neighbors"]),
            }
            for key, fn in score_fns.items():
                m = metrics_mod.evaluate(fn, dataset.train_user_items, dataset.test_user_items,
                                          dataset.num_items, k_list=cfg["k_list"])
                run.results[key] = {"label": METHOD_LABELS[key], "metrics": m}
                run.log(f"[{key}] " + " ".join(f"{mk}={mv:.4f}" for mk, mv in m.items() if mk != "num_eval_users"))

            # --- Diversity ---
            run.current_stage = "diversity"
            run.log("Evaluating diversity (coverage, novelty, intra-list diversity)...")
            for key, fn in score_fns.items():
                dv = diversity_mod.evaluate_diversity(
                    fn, dataset.train_user_items, dataset.test_user_items, dataset.item_genres,
                    item_counts, top_k=cfg["diversity_top_k"])
                run.results[key]["diversity"] = dv
                run.log(f"[{key}] coverage={dv['coverage']:.4f} novelty={dv['novelty']:.4f} "
                         f"ild={dv['intra_list_diversity']:.4f}")

            run.current_stage = "done"
            run.status = "finished"
            run.log("Run finished.")
        except Exception as e:  # pragma: no cover - surfaced to UI, not swallowed
            run.status = "error"
            run.error = f"{type(e).__name__}: {e}"
            run.log(f"ERROR: {run.error}")
        finally:
            run.updated.set()

    # -- on-demand extras ----------------------------------------------------

    def sensitivity_cell(self, run: Run, H: int, tau: float) -> dict:
        key = f"{H}:{tau}"
        with run.lock:
            if key in run.sensitivity_cache:
                return run.sensitivity_cache[key]
        dataset = run._dataset
        cfg = run.config
        model = cgnf_mod.fit(dataset.train_matrix, tau=tau, h_grid=[H], weighted_modularity=True,
                              use_edge_reduction=True, iters=cfg["cgnf_iters"], seed=cfg["seed"])
        score_fn = lambda u: cgnf_mod.score_for_user(model, u, cfg["cgnf_top_k_neighbors"])
        m = metrics_mod.evaluate(score_fn, dataset.train_user_items, dataset.test_user_items,
                                  dataset.num_items, k_list=cfg["k_list"])
        cell = {"H": H, "tau": tau, "modularity": model.modularity, "metrics": m,
                "edges_before": model.edges_before, "edges_after": model.edges_after}
        with run.lock:
            run.sensitivity_cache[key] = cell
        return cell

    def run_multiseed(self, run: Run, num_seeds: int = 5, metric_key: str = "ndcg@20"):
        if run.status != "finished":
            raise ValueError("run must be finished before multi-seed validation")
        cfg = run.config
        best_baseline = max(
            (k for k in run.results if k != "cgnf"),
            key=lambda k: run.results[k]["metrics"].get(metric_key, 0.0))
        run.log(f"Multi-seed validation: {num_seeds} seeds, metric={metric_key}, "
                f"comparing cgnf vs {best_baseline}")

        per_seed = {"cgnf": [], best_baseline: []}
        seeds = list(range(num_seeds))
        for s in seeds:
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            dataset = self._resolve_dataset(cfg_s)

            cgnf_model = cgnf_mod.fit(
                dataset.train_matrix, tau=cfg["cgnf_tau"],
                h_grid=graph_mod.practical_h_grid(dataset.num_users, dataset.num_items, cfg["cgnf_max_h"]),
                weighted_modularity=True, use_edge_reduction=True, iters=cfg["cgnf_iters"], seed=s)
            m_cgnf = metrics_mod.evaluate(
                lambda u: cgnf_mod.score_for_user(cgnf_model, u, cfg["cgnf_top_k_neighbors"]),
                dataset.train_user_items, dataset.test_user_items, dataset.num_items, k_list=cfg["k_list"])
            per_seed["cgnf"].append(m_cgnf.get(metric_key, 0.0))

            if best_baseline == "mostpop":
                bmodel = mostpop.fit(dataset.train_matrix)
                fn = lambda u: bmodel.score_for_user(u)
            elif best_baseline == "itemknn":
                bmodel = itemknn.fit(dataset.train_matrix, k=cfg["itemknn_k"])
                fn = lambda u: bmodel.score_for_user(u)
            elif best_baseline == "bprmf":
                bmodel = bprmf.fit(dataset.train_matrix, embedding_dim=cfg["bprmf_embedding_dim"],
                                    epochs=cfg["bprmf_epochs"], batch_size=cfg["bprmf_batch_size"],
                                    lr=cfg["bprmf_lr"], lr_decay=cfg["bprmf_lr_decay"],
                                    l2_coef=cfg["bprmf_l2_coef"], seed=s)
                fn = lambda u: bmodel.score_for_user(u)
            else:
                bmodel = lightgcn.fit(dataset.train_matrix, embedding_dim=cfg["lightgcn_embedding_dim"],
                                       num_layers=cfg["lightgcn_num_layers"], epochs=cfg["lightgcn_epochs"],
                                       batch_size=cfg["lightgcn_batch_size"], lr=cfg["lightgcn_lr"],
                                       lr_decay=cfg["lightgcn_lr_decay"], l2_coef=cfg["lightgcn_l2_coef"], seed=s)
                fn = lambda u: bmodel.score_for_user(u)
            m_base = metrics_mod.evaluate(fn, dataset.train_user_items, dataset.test_user_items,
                                           dataset.num_items, k_list=cfg["k_list"])
            per_seed[best_baseline].append(m_base.get(metric_key, 0.0))
            run.log(f"[multiseed] seed={s} cgnf={m_cgnf.get(metric_key, 0):.4f} "
                     f"{best_baseline}={m_base.get(metric_key, 0):.4f}")

        aggregated = stats_mod.aggregate(per_seed)
        significance = stats_mod.paired_significance(per_seed["cgnf"], per_seed[best_baseline])
        result = {
            "metric": metric_key, "seeds": seeds, "best_baseline": best_baseline,
            "aggregate": aggregated, "significance": significance,
        }
        run.multiseed = result
        run.log("Multi-seed validation finished.")
        return result


manager = RunManager()
