"""miner_3 2026-08-13: persist close_pos_20 (EFFECTIVE) with full metrics + signal artifact."""
import json, sys, os, datetime
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, N_GRID, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)

def load_asset(sym, days=2200):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df

series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}

fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)

# close_pos_20: mean((close-low)/(high-low), 20)
cp = {}
for s, df in series.items():
    hi, lo, cl = df["high"], df["low"], df["close"]
    rng = (hi - lo).replace(0, np.nan)
    pos = safe_div(cl - lo, rng)
    cp[s] = pd.Series(pos, index=df.index).rolling(20, min_periods=10).mean()
mat = to_grid(cp)

rank_mat = cross_sectional_rank(mat)
ics = spearman_ic_matrix(mat, fwd[10])
summ = summarize(ics, dates, "close_pos_20", HORIZON)
cov_ad, cov_d8 = coverage_stats(mat)
to = turnover_10d_rank(rank_mat)
dec = decay_curve(mat, fwd)
corrs, mx_name, mx_abs = library_pairwise_corr(mat)

ic, icir = summ["ic"], summ["icir"]
print(f"close_pos_20: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
      f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name})")
print("regime:", {k: v for k, v in summ["regime"].items()})
print("decay:", dec)
assert abs(ic) >= 0.007 and abs(icir) >= 0.084, "GATE FAILED - do not persist"

# save signal artifact
np.save("factors/close_pos_20.signal.npy", rank_mat)
os.chmod("factors/close_pos_20.signal.npy", 0o644)

payload = {
    "factor_id": "close_pos_20",
    "factor_name": "Close Position in Daily Range (20d mean) - intraday location / demand pressure",
    "version": "1.0.0",
    "calculation": {
        "expression": "mean_t((close_t - low_t) / (high_t - low_t), 20)  [per asset own calendar; range=0 -> NaN]",
        "description": ("20d average of where each day's close sits inside its daily high-low range. "
                        "High values = closes near daily highs (consistent intraday buying / demand pressure); "
                        "low values = closes near daily lows (selling pressure). Positive 10d rank IC: assets with "
                        "recent strong intraday location continue to outperform over the next 10 days. Distinct from "
                        "pure return momentum because it isolates close-location rather than signed return magnitude."),
        "interpretation": "high value = persistent close-at-high behavior; positive 10d IC"
    },
    "dependencies": ["open", "close", "high", "low"],
    "parameters": {"window": 20, "min_periods": 10, "admission_horizon": 10},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "validated_at": "2026-08-13",
        "last_validated": "2026-08-13",
        "period": "2020-01-01..2026-08-12",
        "admission_horizon": 10,
        "admission_gate": {"ic_abs_min": 0.007, "icir_abs_min": 0.084},
        "metrics": {
            "ic": round(ic, 4),
            "icir": round(icir, 4),
            "ic_hit_ratio": round(summ["hit"], 4),
            "n_ic_dates": summ["n_ic_dates"],
            "coverage_asset_days": round(cov_ad, 4),
            "coverage_dates_ge8": round(cov_d8, 4),
            "n_dates_total": N_GRID,
            "n_dates_ge8": int(((~np.isnan(mat)).sum(axis=1) >= 8).sum()),
            "turnover_10d_rank": round(to, 4),
            "decay_ic_by_horizon": {k: round(v, 4) for k, v in dec.items()},
            "max_abs_library_correlation": round(mx_abs, 4),
            "library_pairwise_corr": {k: v for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:12]},
            "signal_artifact": "factors/close_pos_20.signal.npy"
        },
        "regime_notes": {
            str(k): {"ic": v["ic"], "icir": v["icir"], "n_dates": v["n"]}
            for k, v in summ["regime"].items()
        },
        "universe": "15-instrument tradable cross-asset benchmark (equity indices, commodities, crypto, yields)"
    },
    "tags": ["intraday-location", "demand-pressure", "cross-asset", "10d-horizon"]
}
path = "factors/close_pos_20.json"
with open(path, "w") as f:
    json.dump(payload, f, indent=1, ensure_ascii=False)
os.chmod(path, 0o664)
print("WROTE", path)

# verify readback
chk = json.load(open(path))
assert chk["factor_id"] == "close_pos_20"
assert chk["validation"]["status"] == "EFFECTIVE"
assert abs(chk["validation"]["metrics"]["ic"]) >= 0.007
assert abs(chk["validation"]["metrics"]["icir"]) >= 0.084
assert chk["validation"]["metrics"]["max_abs_library_correlation"] == round(mx_abs, 4)
print("VERIFIED OK: id, status, gates, max_abs_library_correlation, artifact present:",
      os.path.exists(chk["validation"]["metrics"]["signal_artifact"]))
