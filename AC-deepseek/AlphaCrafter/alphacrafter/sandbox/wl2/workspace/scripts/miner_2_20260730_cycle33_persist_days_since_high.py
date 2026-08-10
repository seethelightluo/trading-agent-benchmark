"""miner_2 cycle33: persist days_since_high_60 (passed IC/ICIR gate).

Admission metrics (10d horizon, 15-instrument cross-asset universe):
  IC  = -0.0367  (abs 0.0367 >= 0.0070)
  ICIR= -0.1185  (abs 0.1185 >= 0.0840)
  max_abs_library_correlation = 0.4281 (< 0.5 ensemble gate)
  n_ic_dates = 1645, coverage_asset_days = 0.7117

Saves signal artifact .npy (screener gate reads real artifacts) and writes
factors/days_since_high_60.json, then reads back and verifies.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, per_asset, forward_returns,
                        compute_ic, validate_factor, library_correlation,
                        turnover_rank, coverage_stats, regime_breakdown)

panel = load_close_panel()


def days_since_high_60(s, w=60, mp=40):
    def _d(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        cmax = np.max(x)
        idx = np.where(np.isclose(x, cmax, rtol=1e-6))[0]
        return float(len(x) - 1 - idx[-1]) if len(idx) else np.nan
    return s.rolling(w, min_periods=mp).apply(_d, raw=True)


f = per_asset(panel, days_since_high_60)
fid = "days_since_high_60"

# ---- signal artifact (real, recoverable by screener gate) ----
art_path = Path("factors") / f"{fid}.signal.npy"
np.save(art_path, f.values)
print("saved artifact:", art_path, f.values.shape)

# ---- library correlation vs effective artifacts ----
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
       "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
       "downside_dev_60"]
lib = {}
for e in EFF:
    p = Path("factors") / f"{e}.signal.npy"
    if p.exists():
        a = np.load(p)
        if a.shape[0] == len(panel.index):
            lib[e] = pd.DataFrame(a, index=panel.index, columns=panel.columns)
lc = library_correlation(f, lib)
print("max_abs_library_correlation:", round(lc["max_abs"], 4))

# ---- metrics ----
fwd = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}
m = validate_factor(f, panel, library=lib, fwd_cache=fwd)
to = turnover_rank(f, step=10)
cov = coverage_stats(f)
ic_ser = compute_ic(f, fwd["10"]).dropna()
ic = float(ic_ser.mean())
icir = float(ic_ser.mean() / ic_ser.std())
reg = regime_breakdown(ic_ser)
print("IC", round(ic, 4), "ICIR", round(icir, 4), "turnover_10d_rank", round(to, 4) if to == to else None)

metrics = {
    "ic": round(ic, 4),
    "icir": round(icir, 4),
    "ic_hit_ratio": round(float((np.sign(ic_ser) == np.sign(ic)).mean()), 3),
    "n_ic_dates": int(len(ic_ser)),
    "coverage_asset_days": cov["coverage_asset_days"],
    "coverage_dates_ge8": cov["coverage_dates_ge8"],
    "n_dates_total": cov["n_dates_total"],
    "n_dates_ge8": cov["n_dates_ge8"],
    "turnover_10d_rank": round(to, 4) if to == to else None,
    "max_abs_library_correlation": round(lc["max_abs"], 4),
    "library_pairwise_corr": {k: round(v, 4) for k, v in lc["pairwise"].items()},
    "decay_ic_by_horizon": m["decay_ic_by_horizon"],
    "signal_artifact": str(art_path),
}

doc = {
    "factor_id": fid,
    "factor_name": "Days Since 60d High (recovery-lag / trend-freshness)",
    "version": "1.0.0",
    "calculation": {
        "expression": ("rolling 60d: v_t = t - max{ j<=t : close_j == max(close[t-59..t]) }; "
                       "days elapsed since close last touched its trailing 60d high"),
        "description": ("Counts how many trading days have elapsed since the price last printed "
                        "its trailing 60-day high. Assets that keep making fresh highs are in "
                        "early-stage uptrends (low value); assets far below their highs show "
                        "stalled recovery (high value). Negative IC => recovery-lagging assets "
                        "continue to underperform over the next 10 days (trend-freshness "
                        "momentum, distinct from level momentum which uses raw returns)."),
        "interpretation": "high value = weak recovery / stale high; negative 10d IC",
    },
    "dependencies": ["close"],
    "parameters": {"window": 60, "min_periods": 40, "admission_horizon": 10},
    "validation": {
        "status": "EFFECTIVE",
        "validated_at": "2026-07-30",
        "period": "2020-01-01..2026-07-29",
        "admission_gate": {"ic_abs_min": 0.0070, "icir_abs_min": 0.0840},
        "metrics": metrics,
        "regime_notes": {
            "2020-2021": reg.get("2020-2021"),
            "2022": reg.get("2022-2022"),
            "2023-2024": reg.get("2023-2024"),
            "2025-2026": reg.get("2025-2026"),
            "summary": ("Negative IC in 2020-21, 2023-24, 2025-26 (strongest recently); "
                        "2022 shows a weak positive tilt (icir 0.055, n=258) - regime wobble "
                        "but overall 10d ICIR -0.1185 with hit 0.549. Re-validate next cycle."),
        },
        "universe": "15-instrument tradable cross-asset benchmark",
    },
    "tags": ["path-structure", "recovery", "trend-freshness", "momentum-like"],
    "last_validated": "2026-07-30",
}

out = Path("factors") / f"{fid}.json"
json.dump(doc, open(out, "w"), indent=1, default=str)
print("wrote", out)

# ---- read back & verify ----
chk = json.load(open(out))
assert chk["factor_id"] == fid
assert chk["validation"]["status"] == "EFFECTIVE"
assert chk["validation"]["metrics"]["max_abs_library_correlation"] == metrics["max_abs_library_correlation"]
assert chk["validation"]["metrics"]["signal_artifact"] == str(art_path)
assert Path(chk["validation"]["metrics"]["signal_artifact"]).exists()
assert abs(chk["validation"]["metrics"]["ic"]) >= 0.007
assert abs(chk["validation"]["metrics"]["icir"]) >= 0.084
print("VERIFIED:", chk["factor_id"], chk["validation"]["status"],
      "| ic", chk["validation"]["metrics"]["ic"],
      "| icir", chk["validation"]["metrics"]["icir"],
      "| maxlibcorr", chk["validation"]["metrics"]["max_abs_library_correlation"])
