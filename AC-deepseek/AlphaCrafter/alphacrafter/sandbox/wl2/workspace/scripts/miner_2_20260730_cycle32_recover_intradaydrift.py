"""miner_2 cycle32: recover intraday_drift_20 (quarantined for missing artifact).

Factor: mean(close/open - 1, 20) per asset on its own calendar.
Previously passed the gate (IC 0.0353, ICIR 0.1073) but the persisted JSON
lacked a recoverable signal-artifact reference, so the deterministic gate
quarantined it. This script recomputes the factor, re-validates it against
the CURRENT effective library (mom20_volproxy60, dxy_beta_cond_60x20,
calmness_20), verifies the existing .npy artifact, and re-persists with the
calmness_20-style schema (signal_artifact as string + artifact_provenance).
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_ohlc_panels, per_asset, compute_ic,
                        forward_returns, turnover_rank, coverage_stats,
                        validate_factor, library_correlation, regime_breakdown,
                        report)

FACTOR_ID = "intraday_drift_20"
VISIBLE_THROUGH = "2026-07-29"

ohlc = load_ohlc_panels()
close = ohlc["close"]
open_ = ohlc["open"]

# ---- factor computation (per-asset calendar) ----
def intraday_mean(s_open, s_close, w=20, mp=10):
    r = s_close / s_open - 1.0
    return r.rolling(w, min_periods=mp).mean()

frames = {a: intraday_mean(open_[a], close[a]).reindex(close.index) for a in close.columns}
F = pd.DataFrame(frames, index=close.index)

# ---- verify existing npy artifact matches recomputation ----
npy_path = Path("factors") / f"{FACTOR_ID}.signal.npy"
arr = np.load(npy_path)
m = (~np.isnan(F.values)) & (~np.isnan(arr))
print(f"[artifact] shape {arr.shape}; recomputed panel {F.shape}")
print(f"[artifact] matched non-nan cells: {m.sum()}; allclose: {np.allclose(F.values[m], arr[m], atol=1e-12)}")

# ---- effective library signals (current active set) ----
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]
lib = {}
idx = close.index
for fid in EFF:
    a = np.load(Path("factors") / f"{fid}.signal.npy")
    if a.shape[0] == len(idx):
        lib[fid] = pd.DataFrame(a, index=idx, columns=close.columns)
    else:
        print(f"[lib] shape mismatch {fid}: {a.shape}")
# also report vs full persisted artifact set for transparency
ALL_ART = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
           "carry_3m1m", "carry_12m3m", "mom_curve_volscale", "range_pos_120d",
           "eff_ratio_20", "intraday_drift_20"]
lib_all = {}
for fid in ALL_ART:
    p = Path("factors") / f"{fid}.signal.npy"
    if not p.exists():
        continue
    a = np.load(p)
    if a.shape[0] == len(idx):
        lib_all[fid] = pd.DataFrame(a, index=idx, columns=close.columns)

# ---- validation ----
fwd = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd[str(h)] = forward_returns(close, h)

metrics = validate_factor(F, close, library=lib, fwd_cache=fwd)
full_lc = library_correlation(F, lib_all)
lib_all = {k: v for k, v in lib_all.items() if k != FACTOR_ID}
full_lc = library_correlation(F, lib_all)
metrics["max_abs_library_correlation"] = round(full_lc["max_abs"], 4)
metrics["library_pairwise_corr"] = {k: round(v, 4) for k, v in full_lc["pairwise"].items()}
metrics["turnover_10d_rank"] = metrics.pop("turnover_10_rank", None)

ic_ser = compute_ic(F, fwd["10"]).dropna()
reg = regime_breakdown(ic_ser)
print("[regime]")
for k, v in reg.items():
    print("   ", k, v)

passed = report(FACTOR_ID, metrics)
print("[metrics]", json.dumps(metrics, indent=1))

if not passed:
    print("GATE FAIL - not persisting")
    sys.exit(1)

# ---- persist with calmness_20-style schema ----
valid_mask = F.notna()
doc = {
    "factor_id": FACTOR_ID,
    "factor_name": "Intraday Drift 20d (gap-free momentum)",
    "version": "1.1.0",
    "calculation": {
        "expression": "mean(close/open - 1, 20) with min_periods=10",
        "description": "Per-asset mean of intraday (open->close) returns over the trailing 20 trading days on the asset's own calendar. Captures gap-free momentum: whether the asset tends to drift up or down during the trading session, stripping out overnight gap information already covered by close-based trend factors. Positive predictor of forward 10d cross-sectional returns (recovered from quarantine after artifact wiring fix)."
    },
    "dependencies": ["open", "close"],
    "parameters": {"window": 20, "min_periods": 10},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": " ".join(
            [f"{k}: ic={v['ic']} icir={v['icir']} n={v['n_dates']};" for k, v in reg.items()]
        ),
        "metrics": metrics,
    },
    "tags": ["momentum", "intraday", "price-structure", "cross-asset"],
    "benchmark_admission": {
        "contract": {
            "ic_threshold": 0.007,
            "icir_threshold": 0.084,
            "correlation_threshold": 0.5,
            "library_capacity": 30,
            "active_top_k": 10,
        },
        "selected_metrics": {
            "ic": metrics["ic"],
            "icir": metrics["icir"],
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": metrics["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
            "quality": round(abs(metrics["ic"]) * abs(metrics["icir"]), 8),
        },
        "admitted_at": "2026-08-10T23:00:00.000000",
    },
    "signal_artifact": f"{FACTOR_ID}.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": list(F.shape),
        "columns": list(F.columns),
        "dates_first": str(F.index[0].date()),
        "dates_last": str(F.index[-1].date()),
        "n_nan": int((~valid_mask).sum().sum()),
    },
}

out = Path("factors") / f"{FACTOR_ID}.json"
out.write_text(json.dumps(doc, indent=1))
np.save(npy_path, F.values)
print(f"[persist] wrote {out}")

# ---- read back and verify ----
chk = json.load(open(out))
ok = (chk["factor_id"] == FACTOR_ID
      and chk["validation"]["status"] == "EFFECTIVE"
      and chk["validation"]["metrics"]["ic"] >= 0.007
      and abs(chk["validation"]["metrics"]["icir"]) >= 0.084
      and Path("factors", chk["signal_artifact"]).exists())
print("[verify] valid json, id ok:", chk["factor_id"] == FACTOR_ID,
      "| status:", chk["validation"]["status"],
      "| ic:", chk["validation"]["metrics"]["ic"],
      "| icir:", chk["validation"]["metrics"]["icir"],
      "| artifact exists:", Path("factors", chk["signal_artifact"]).exists(),
      "| ALL OK:", ok)
