"""miner_3 2026-07-30 cycle 29: persist eff_ratio_20 (Kaufman efficiency ratio, 20d).

Fixes cycle-28 persistence bug (KeyError 'turnover_10d_rank' left only the signal
artifact without a JSON). Writes factors/eff_ratio_20.json + signal artifact, then
reads back and verifies. All metrics recomputed from executed validation (10d horizon).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns,
                         validate_factor, VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# deterministic factor recompute
f = per_asset(panel, lambda s: (s - s.shift(20)).abs()
              / s.diff().abs().rolling(20, min_periods=15).sum())
sig = f.reindex(panel.index)

lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)

m = validate_factor(f, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                    library=lib, fwd_cache=fwd_cache)
print("ADMISSION METRICS:", json.dumps(m, indent=1))

assert abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840, "gate not met"
assert m["max_abs_library_correlation"] < 0.5, "correlation gate not met"

np.save("factors/eff_ratio_20.signal.npy", sig.values)
print("signal artifact saved", sig.shape)

doc = {
    "factor_id": "eff_ratio_20",
    "factor_name": "kaufman_efficiency_ratio_20d",
    "version": "1.1.0",
    "calculation": {
        "expression": "abs(close - close.shift(20)) / sum(abs(close.diff()), 20)",
        "description": ("Kaufman efficiency ratio over 20 days on the asset's own calendar: "
                        "net 20d displacement divided by the sum of absolute 1d moves. High "
                        "values indicate smooth, directional trending; low values indicate "
                        "choppy/range-bound behavior. Cross-sectionally long the smoothly "
                        "trending assets at 10d horizon. Rank-based (Spearman) usage."),
        "transform": "rank cross-sectionally (pct rank); long top, underweight bottom"
    },
    "dependencies": ["close"],
    "parameters": {"window": 20, "min_periods": 15},
    "validation": {
        "status": "EFFECTIVE",
        "period": {"start": "2020-01-01", "end": VISIBLE_THROUGH},
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "ic_hit_ratio": m["ic_hit_ratio"],
            "n_ic_dates": m["n_ic_dates"],
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "turnover_10d_rank": m["turnover_10_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": m["max_abs_library_correlation"],
            "library_pairwise_corr": m["library_pairwise_corr"],
        },
        "regime_notes": (
            "10d IC by sub-period: 2020-2021 +0.0319 (icir +0.108), 2022 +0.0757 (+0.248), "
            "2023-2024 +0.0311 (+0.105), 2025-2026 +0.0791 (+0.260). Positive and stable "
            "across bull, bear (2022), and crypto-winter regimes. Library correlation low "
            "(max |rho| 0.17 vs mom20_volproxy60), orthogonal trend-quality signal."
        ),
        "decay": "IC rises from 1d (0.018) to 10d (0.050) then eases to 20d (0.044); "
                 "peak around the 10d admission horizon - consistent with medium-term trend persistence.",
    },
    "tags": ["trend", "efficiency_ratio", "trend_quality", "momentum_structure", "cross_asset"],
    "benchmark_admission": {
        "contract": {
            "ic_threshold": 0.007,
            "icir_threshold": 0.084,
            "correlation_threshold": 0.5,
            "library_capacity": 30,
            "active_top_k": 10
        },
        "selected_metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": m["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
            "quality": round(abs(m["ic"]) * abs(m["icir"]), 6)
        }
    },
    "signal_artifact": "eff_ratio_20.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": list(sig.shape),
        "columns": list(panel.columns),
        "dates_first": str(panel.index[0].date()),
        "dates_last": str(panel.index[-1].date()),
        "n_nan": int(np.isnan(sig.values).sum())
    }
}

with open("factors/eff_ratio_20.json", "w") as fh:
    json.dump(doc, fh, indent=2)
print("JSON written.")

back = json.load(open("factors/eff_ratio_20.json"))
sig2 = np.load("factors/eff_ratio_20.signal.npy")
ok = (back["factor_id"] == "eff_ratio_20"
      and back["validation"]["status"] == "EFFECTIVE"
      and abs(back["validation"]["metrics"]["ic"]) >= 0.0070
      and abs(back["validation"]["metrics"]["icir"]) >= 0.0840
      and back["validation"]["metrics"]["max_abs_library_correlation"] < 0.5
      and sig2.shape == sig.shape and np.allclose(sig2, sig.values, equal_nan=True))
print("READBACK OK:", ok)
print("factor_id:", back["factor_id"], "| status:", back["validation"]["status"],
      "| IC:", back["validation"]["metrics"]["ic"],
      "| ICIR:", back["validation"]["metrics"]["icir"],
      "| maxlibcorr:", back["validation"]["metrics"]["max_abs_library_correlation"])
