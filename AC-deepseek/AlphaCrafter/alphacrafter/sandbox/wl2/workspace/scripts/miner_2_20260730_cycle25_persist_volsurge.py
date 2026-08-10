"""miner_2 cycle 25 persistence: vol_surge_20 (volume surge factor).

Passed the shared admission gate (IC=0.0586 >= 0.007, ICIR=0.135 >= 0.084,
maxlibcorr=0.1101 < 0.5). Persist full definition + signal artifact, then
read back and verify.
"""
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
import pandas as pd
from miner2_lib import (load_close_panel, load_volume_panel, per_asset,
                        validate_factor, load_library_signals, forward_returns,
                        save_signal_artifact, regime_breakdown, compute_ic,
                        TRADABLES, VISIBLE_THROUGH)

FID = "vol_surge_20"
panel = load_close_panel()
vol = load_volume_panel()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}

# --- recompute factor exactly as screened ---
f = per_asset(vol, lambda s: (s / s.rolling(20, min_periods=10).mean() - 1.0))
m = validate_factor(f, panel, library=lib, fwd_cache=fwd_cache)
print("admission metrics:", {k: m[k] for k in ("ic", "icir", "ic_hit_ratio",
      "n_ic_dates", "coverage_asset_days", "coverage_dates_ge8",
      "max_abs_library_correlation", "decay_ic_by_horizon")})

# regime breakdown on admission-horizon IC
ret10 = fwd_cache["10"]
ic_ser = compute_ic(f, ret10).dropna()
regimes = regime_breakdown(ic_ser)
print("regime breakdown:", regimes)

# --- persist signal artifact (real gate input) ---
art = save_signal_artifact(FID, f)
print("artifact saved:", art, f.values.shape)

# --- build factor JSON in the shared schema ---
n_nan = int(np.isnan(f.values).sum())
doc = {
    "factor_id": FID,
    "factor_name": "20-day volume surge",
    "version": "1.0.0",
    "calculation": {
        "expression": "volume / SMA20(volume, min_periods=10) - 1",
        "description": ("Per-asset 20-day rolling mean volume normalization: how much "
                        "today's volume exceeds (or falls below) its trailing 20-day "
                        "average. Captures participation/liquidity regime shifts "
                        "orthogonal to close-price factors; positive IC means volume "
                        "surges tend to precede higher 10-day forward returns.")
    },
    "dependencies": ["volume"],
    "parameters": {"window": 20, "min_periods": 10},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": f"2020-01-01..{VISIBLE_THROUGH}",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": ("15-instrument tradable cross-asset universe; volume series "
                         "present for all 15 assets but sparse before ~2021 (crypto "
                         "full-history). ") +
                        "; ".join(f"{k}: ic={v['ic']} icir={v['icir']} n={v['n_dates']}"
                                  for k, v in regimes.items()),
        "metrics": {
            **{k: m[k] for k in ("ic", "icir", "ic_hit_ratio", "n_ic_dates",
                                 "coverage_asset_days", "coverage_dates_ge8",
                                 "turnover_10d_rank", "decay_ic_by_horizon",
                                 "max_abs_library_correlation",
                                 "library_pairwise_corr")},
            "regime_breakdown": regimes,
        },
    },
    "tags": ["volume", "liquidity", "participation", "flow"],
    "benchmark_admission": {
        "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                     "correlation_threshold": 0.5, "library_capacity": 30,
                     "active_top_k": 10},
        "selected_metrics": {
            "ic": m["ic"], "icir": m["icir"],
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": m["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
            "quality": round(abs(m["ic"]) * abs(m["icir"]), 8),
        },
        "admitted_at": pd.Timestamp.now().isoformat(),
    },
    "signal_artifact": f"{FID}.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": list(f.values.shape),
        "columns": TRADABLES,
        "dates_first": str(panel.index[0].date()),
        "dates_last": str(panel.index[-1].date()),
        "n_nan": int(n_nan),
    },
}

out_path = Path("factors") / f"{FID}.json"
out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
print("wrote", out_path)

# --- read back and verify ---
back = json.loads(out_path.read_text())
checks = {
    "valid_json": True,
    "factor_id": back.get("factor_id") == FID,
    "status": back.get("validation", {}).get("status") == "EFFECTIVE",
    "ic_gate": abs(back["validation"]["metrics"]["ic"]) >= 0.007,
    "icir_gate": abs(back["validation"]["metrics"]["icir"]) >= 0.084,
    "artifact_exists": Path("factors", back["signal_artifact"]).exists(),
    "artifact_shape": list(np.load(Path("factors", back["signal_artifact"])).shape) == list(f.values.shape),
}
print("verify:", checks)
assert all(checks.values()), checks
print("PERSIST_OK", FID)
