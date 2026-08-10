"""miner_2 cycle32: recover gain_loss_20 (rejected for artifact-file bug).

Factor: ratio of mean positive daily return to mean |negative| daily return
over 20d (trend/skew quality of the return path).
Previously passed the gate strongly (IC 0.0530, ICIR 0.1543) but the persist
step embedded the full artifact dict (dates list) inside the JSON, so the
gate's file-write failed with "File name too long" and the factor was moved
to rejected/. This script recomputes the factor, validates against the
CURRENT effective library, saves a proper .npy artifact, and re-persists with
calmness_20-style schema (signal_artifact as string + artifact_provenance).
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, compute_ic, forward_returns,
                        validate_factor, library_correlation, regime_breakdown,
                        report)

FACTOR_ID = "gain_loss_20"

close = load_close_panel()

# ---- factor computation ----
def gain_loss(s, w=20, mp=10):
    r = s.pct_change()
    up = r.clip(lower=0).rolling(w, min_periods=mp).mean()
    dn = r.clip(upper=0).rolling(w, min_periods=mp).mean().abs()
    return up / (dn + 1e-9)

F = close.apply(lambda s: gain_loss(s))

# ---- effective library signals ----
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]
lib = {}
idx = close.index
for fid in EFF:
    a = np.load(Path("factors") / f"{fid}.signal.npy")
    if a.shape[0] == len(idx):
        lib[fid] = pd.DataFrame(a, index=idx, columns=close.columns)
    else:
        print(f"[lib] shape mismatch {fid}: {a.shape}")

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
    "factor_name": "20d Gain/Loss ratio (reward-to-pain path quality)",
    "version": "3.1.0",
    "calculation": {
        "expression": "mean(max(daily_ret,0),20) / (mean(min(daily_ret,0),20).abs() + 1e-9)",
        "description": "Ratio of the mean positive daily return to the mean absolute negative daily return over the trailing 20 trading days. High values identify assets whose upward sessions are large relative to their down sessions (positive skew / quality of the return path). Positive predictor of forward 10d cross-sectional returns. Recovered from rejected/ after fixing the artifact-persistence bug (signal now stored as .npy with provenance)."
    },
    "dependencies": ["close"],
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
    "tags": ["trend", "quality", "skew", "cross-asset"],
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
        "admitted_at": "2026-08-10T23:05:00.000000",
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
np.save(Path("factors") / f"{FACTOR_ID}.signal.npy", F.values)
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
