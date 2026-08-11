"""miner_2 persist amihud_20 with signal artifact (npy) so it passes the
deterministic pairwise gate (worldline_pairwise_signal_quality_v1).
Universe: 15 tradable cross-asset instruments. Validation window 2020-01-01..2026-07-15.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
import miner_2_lib as lib

EPS = 1e-12
WATCH = lib.WATCH
MAX_VISIBLE = lib.MAX_VISIBLE
FACTOR_LAST = lib.FACTOR_LAST
MIN_ASSETS = lib.MIN_ASSETS
ADMISSION = lib.ADMISSION

panel = lib.load_panel()

# ---- factor: Amihud illiquidity, 20d mean of |ret|/volume (per-asset calendar) ----
factor = {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
    r = df["close"].pct_change()
    factor[s] = (r.abs() / (df["volume"] + EPS)).rolling(20, min_periods=10).mean()
factor = pd.DataFrame(factor, index=panel.index)
factor_w = factor.loc[:FACTOR_LAST]
print("factor dates:", factor.index.min().date(), "..", factor.index.max().date(), len(factor))
print("warmup rows:", len(factor_w))


def fwd_returns(h):
    out = {}
    for s in WATCH:
        c = panel[s].dropna()
        out[s] = c.shift(-h) / c - 1.0
    return pd.DataFrame(out, index=panel.index)


def rank_ic(f, fwd):
    ics = {}
    for d in f.index.intersection(fwd.index):
        fv = f.loc[d].dropna()
        rv = fwd.loc[d].reindex(fv.index).dropna()
        if len(rv) < MIN_ASSETS:
            continue
        ics[d] = spearmanr(fv.reindex(rv.index), rv)[0]
    return pd.Series(ics).sort_index()


horizons = (1, 2, 3, 5, 10, 20)
fwd = {h: fwd_returns(h) for h in horizons}
ic_by_h = {h: rank_ic(factor_w, fwd[h]) for h in horizons}
res = {}
direction = 1.0  # raw h10 IC is positive
for h in horizons:
    ic = ic_by_h[h] * direction
    res[f"ic_h{h}"] = float(ic.mean())
    res[f"icir_h{h}"] = float(ic.mean() / ic.std())
    res[f"hit_h{h}"] = float((ic > 0).mean())
    res[f"n_dates_h{h}"] = int(len(ic))

valid = factor_w.notna()
res["coverage_asset_days"] = float(valid.mean().mean())
res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())

ranks = factor_w.rank(axis=1)
tov = []
for i in range(10, len(ranks)):
    a, b = ranks.iloc[i - 10], ranks.iloc[i]
    both = a.dropna().index.intersection(b.dropna().index)
    if len(both) >= MIN_ASSETS:
        tov.append(float((a[both] - b[both]).abs().mean()))
res["turnover_10d_rank"] = float(np.mean(tov))

# per-year robustness
years = {}
for y in range(2020, 2027):
    sub = ic_by_h[10].loc[str(y)]
    if len(sub) > 20:
        years[y] = {"ic": round(float(sub.mean()), 4),
                    "icir": round(float(sub.mean() / sub.std()), 4),
                    "n": int(len(sub))}
print("per-year h10 IC:", years)

# library correlation (same as lib over last 700 dates)
libs = lib.library_signals(panel)
per = {}
for fid, lf in libs.items():
    cs = []
    for dt in factor.index[-700:]:
        if dt not in lf.index:
            continue
        f = factor.loc[dt]
        g = lf.loc[dt]
        m = f.notna() & g.notna()
        m = m.reindex(f.index).fillna(False)
        if int(m.sum()) >= MIN_ASSETS:
            cs.append(spearmanr(f[m], g[m])[0])
    per[fid] = round(float(np.mean(cs)), 4) if cs else None
max_corr = round(max([abs(v) for v in per.values() if v is not None]), 4)
print("library corr:", per, "max:", max_corr)

gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
print(f"ADMISSION h10: |IC|={abs(res['ic_h10']):.4f} {gate_ic}, "
      f"|ICIR|={abs(res['icir_h10']):.4f} {gate_icir} -> "
      f"{'PASS' if gate_ic and gate_icir else 'FAIL'}")
assert gate_ic and gate_icir, "does not pass admission gate"

# ---- signal artifact (full visible window, union calendar) ----
artifact = np.asarray(factor.reindex(panel.index).values, dtype=float)  # (n_dates, 15)
npy_path = Path("factors/amihud_20.signal.npy")
np.save(npy_path, artifact)
print("artifact saved:", npy_path, artifact.shape, "n_nan:", int(np.isnan(artifact).sum()))

# ---- persist JSON ----
payload = {
    "factor_id": "amihud_20",
    "factor_name": "Amihud Illiquidity 20d (cross-asset)",
    "version": "1.0.0",
    "calculation": {
        "expression": "rolling_mean(|pct_change(close)| / volume, 20)",
        "description": "20-day average of absolute daily return divided by volume (Amihud illiquidity proxy) "
                       "computed on each instrument's own trading calendar. Higher values identify assets whose "
                       "price moves occur on thin volume (less liquid / more fragile); in this 15-instrument "
                       "cross-asset benchmark high illiquidity positively predicts forward 10-day cross-sectional "
                       "returns (liquidity-risk premium).",
    },
    "dependencies": ["close", "volume"],
    "parameters": {"window": 20, "min_periods": 10, "volume_floor": 1e-12},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-15",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": (
            "15-instrument tradable cross-asset universe (equity idx, commodities, crypto, yields). "
            "Per-year h10 IC: " + "; ".join(f"{y}: ic={v['ic']} icir={v['icir']} n={v['n']}" for y, v in sorted(years.items()))
        ),
        "metrics": {
            "ic": res["ic_h10"],
            "icir": res["icir_h10"],
            "ic_hit_ratio": res["hit_h10"],
            "n_ic_dates": res["n_dates_h10"],
            "coverage_asset_days": res["coverage_asset_days"],
            "coverage_dates_ge8": res["coverage_dates_ge8"],
            "turnover_10d_rank": res["turnover_10d_rank"],
            "decay_ic_by_horizon": {str(h): round(res[f"ic_h{h}"], 4) for h in horizons},
            "max_abs_library_correlation": max_corr,
            "library_pairwise_corr": {k: v for k, v in per.items() if v is not None},
        },
    },
    "tags": ["liquidity", "amihud", "cross-asset", "volume"],
    "signal_artifact": "amihud_20.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": list(artifact.shape),
        "columns": WATCH,
        "dates_first": str(panel.index.min().date()),
        "dates_last": str(panel.index.max().date()),
        "n_nan": int(np.isnan(artifact).sum()),
    },
    "benchmark_admission": {
        "contract": {"ic_threshold": ADMISSION["ic"], "icir_threshold": ADMISSION["icir"],
                     "correlation_threshold": 0.5},
        "selected_metrics": {
            "ic": res["ic_h10"],
            "icir": res["icir_h10"],
            "metric_path": "validation.metrics",
            "max_abs_library_correlation": max_corr,
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
    },
}
path = Path("factors/amihud_20.json")
path.write_text(json.dumps(payload, indent=2))
print("PERSISTED ->", path)

# ---- verify reload ----
loaded = json.loads(path.read_text())
assert loaded["factor_id"] == "amihud_20"
assert loaded["validation"]["status"] == "EFFECTIVE"
assert loaded["validation"]["metrics"]["max_abs_library_correlation"] == max_corr
arr = np.load(npy_path)
assert arr.shape == tuple(loaded["artifact_provenance"]["shape"])
print("VERIFIED: valid JSON, id ok, status EFFECTIVE, artifact reloadable "
      f"shape={arr.shape} nan={int(np.isnan(arr).sum())}")
