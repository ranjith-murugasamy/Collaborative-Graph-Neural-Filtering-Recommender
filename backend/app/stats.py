"""Multi-seed statistical validation — Reviewer 2 item 5: "Report results
from multiple runs with different random seeds, including mean ± standard
deviation/confidence intervals and significance testing." Unlike the
sibling hybrid-movie-recommender project (where this was named as future
work and left unimplemented), it's built here.

Pairing is per-seed: the same N seeds are used for every method being
compared, so seed i's CGNF result and seed i's baseline result are a
genuine paired observation (both trained/evaluated on the same held-out
split, differing only in the model's own random initialization/sampling).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy is a declared dependency; defensive only
    _scipy_stats = None


def aggregate(per_seed_values: Dict[str, List[float]]) -> Dict[str, dict]:
    """per_seed_values: {method_name: [metric_value_per_seed, ...]}"""
    out = {}
    for method, values in per_seed_values.items():
        arr = np.asarray(values, dtype=np.float64)
        out[method] = {
            "seeds": len(arr),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "values": arr.tolist(),
        }
    return out


def paired_significance(values_a: List[float], values_b: List[float]) -> dict:
    """Paired comparison between two methods' per-seed values (same seeds,
    same split). Reports a paired t-test always (valid for small N under
    an approx-normal assumption of the per-seed means) and a Wilcoxon
    signed-rank test when there are enough non-tied pairs for scipy to
    compute it meaningfully (conventionally N >= 6)."""
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    if _scipy_stats is None or len(a) < 2 or len(a) != len(b):
        return {"available": False, "reason": "need >=2 paired seed values"}

    result = {"available": True, "n_pairs": len(a), "mean_diff": float(np.mean(a - b))}
    try:
        t_stat, t_p = _scipy_stats.ttest_rel(a, b)
        result["paired_t_test"] = {"statistic": float(t_stat), "p_value": float(t_p)}
    except Exception as e:  # pragma: no cover
        result["paired_t_test"] = {"error": str(e)}

    diffs_nonzero = np.count_nonzero(a - b)
    if diffs_nonzero >= 6:
        try:
            w_stat, w_p = _scipy_stats.wilcoxon(a, b)
            result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p)}
        except Exception as e:  # pragma: no cover
            result["wilcoxon"] = {"error": str(e)}
    else:
        result["wilcoxon"] = {
            "available": False,
            "reason": f"only {diffs_nonzero} non-tied pairs; conventionally need >=6",
        }
    return result
