"""Persist calmness_20 (Cycle 30 Family A winner).

Gate: |IC|=0.0292 >= 0.007, |ICIR|=0.0997 >= 0.084, maxlibcorr=0.0466 < 0.5
vs active library (mom20_volproxy60, dxy_beta_cond_60x20).
Writes signal artifact + JSON, then verifies by reading back.
"""
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor)

panel = load_panel()
close = panel

# ---- Rebuild active library for correlation report ----
mom60_proxy = per_asset(close, lambda s: s.shift(5) / s.shift(65) - 1.0)
damp = 1.0 / (1.0 + mom60_proxy.abs())
sig_mom = per_asset(close, lambda s: s.shift(5) / s.shift(25) - 1.0) * damp
dxy = macro_series("DXY")
dxy_ret = dxy.pct_change()
dxy_20 = dxy / dxy.shift(20) - 1.0
beta_parts = {}
for a in close.columns:
    s = close[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), dxy_ret.reindex(ar.index).rename("d")], axis=1).dropna()
    b = df["a"].rolling(60).cov(df["d"]) / df["d"].rolling(60).var()
    beta_parts[a] = b.reindex(panel.index)
beta_panel = pd.DataFrame(beta_parts, index=panel.index)
sig_dxy = beta_panel.mul(dxy_20.reindex(beta_panel.index), axis=0)
library = {"mom20_volproxy60": sig_mom, "dxy_beta_cond_60x20": sig_dxy}

# ---- Calmness signal: fraction of last 20d with |ret| < 0.5 * 20d std ----
def calmness_20(s):
    return s.pct_change().abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan,
        raw=True)

sig = per_asset(close, calmness_20)

# ---- Full validation ----
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)
m = validate_factor(sig, panel, library=library, fwd_cache=fwd_cache)
print("FULL VALIDATION:", json.dumps(m, indent=1))

# ---- Regime (per-year) splits for regime_notes ----
ret10 = fwd_cache["10"]
ic_ser = compute_ic(sig, ret10, 8).dropna()
years = ic_ser.index.year
regime_parts = []
for y in sorted(set(years)):
    sub = ic_ser[years == y]
    regime_parts.append(f"{y}: ic={sub.mean():.4f} icir={(sub.mean()/sub.std()) if sub.std()>0 else 0:.4f} n={len(sub)}")
regime_notes = "15-instrument tradable cross-asset universe; " + "; ".join(regime_parts)

# ---- Persist signal artifact (2398x15, TRADABLES order) ----
sig_matrix = sig.values.astype(np.float64)
n_nan = int(np.isnan(sig_matrix).sum())
np.save("factors/calmness_20.signal.npy", sig_matrix)

factor_id = "calmness_20"
doc = {
    "factor_id": factor_id,
    "factor_name": "Calmness 20d (quiet-regime persistence)",
    "version": "1.0.0",
    "calculation": {
        "expression": "rolling_mean(|daily_ret| < 0.5 * rolling_std(daily_ret, 20), 20)",
        "description": "Fraction of the trailing 20 trading days on which the absolute daily return was below half the 20d standard deviation. High values identify assets in persistent low-volatility/quiet regimes (low-vol anomaly); low values identify assets with frequent outsized moves. Positive predictor of forward 10d cross-sectional returns."
    },
    "dependencies": ["close"],
    "parameters": {
        "window": 20,
        "threshold_multiple": 0.5,
        "min_periods": 10
    },
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": regime_notes,
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
            "library_pairwise_corr": m["library_pairwise_corr"]
        }
    },
    "tags": ["volatility", "low-vol", "regime", "cross-asset"],
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
            "quality": round(abs(m["ic"]) * abs(m["icir"]), 8)
        },
        "admitted_at": "2026-08-10T22:15:00"
    },
    "signal_artifact": "calmness_20.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": [int(sig_matrix.shape[0]), int(sig_matrix.shape[1])],
        "columns": list(close.columns),
        "dates_first": str(panel.index.min().date()),
        "dates_last": str(panel.index.max().date()),
        "n_nan": n_nan
    }
}

out = f"factors/{factor_id}.json"
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
print("WROTE", out)

# ---- Verify read-back ----
chk = json.load(open(out))
assert chk["factor_id"] == factor_id, "id mismatch"
assert chk["validation"]["status"] == "EFFECTIVE", "status"
assert chk["validation"]["metrics"]["ic"] == m["ic"]
assert chk["validation"]["metrics"]["icir"] == m["icir"]
art = np.load("factors/calmness_20.signal.npy")
assert art.shape == (2398, 15), f"artifact shape {art.shape}"
assert chk["benchmark_admission"]["contract"]["ic_threshold"] == 0.007
assert chk["benchmark_admission"]["contract"]["icir_threshold"] == 0.084
print("VERIFY OK: json reloadable, id/status/metrics/artifact consistent")
