"""miner_3 2026-07-30 cycle 28: explore trend-quality & return-structure factor family.

Motivation: the library currently holds only vol-damped momentum (mom20_volproxy60) and
DXY-beta conditioning (dxy_beta_cond_60x20). Raw momentum windows, carry, range position
and vol-of-vol families were evicted at the |rho|<0.5 correlation gate. We explore
constructs that measure *how* a trend is formed (smoothness, acceleration, vol term
structure, autocorrelation, tail shape, gap behavior) rather than raw direction, hoping
for orthogonal predictive signal at the 10d admission horizon.

All factors are computed per-asset on the asset's own calendar (no NaN gaps), then
reindexed to the union panel for cross-sectional IC. Visible cutoff 2026-07-29.
Library = the two ACTIVE persisted factors (loaded from real signal artifacts).
Admission gate: |IC| >= 0.0070 AND |ICIR| >= 0.0840 (15-instrument benchmark).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, report, VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# --- library = ACTIVE persisted factors (real artifacts) ---
lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
print(f"library loaded: {list(lib.keys())}; panel {panel.shape} "
      f"dates {panel.index.min().date()}..{panel.index.max().date()}")

# load OHLC for gap factor
def load_ohlc(field):
    out = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        out[a] = pd.Series(df[field].astype(float).values, index=pd.to_datetime(df["date"]), name=a)
    return pd.DataFrame(out, index=panel.index).sort_index()

open_p = load_ohlc("open")
high_p = load_ohlc("high")
low_p = load_ohlc("low")

# ---------------------------------------------------------------------------
# Candidate constructions (single-idea family: trend quality & structure)
# ---------------------------------------------------------------------------
# 1) efficiency_ratio_20: Kaufman efficiency ratio, 20d
f_er20 = per_asset(panel, lambda s: (s - s.shift(20)).abs() / s.diff().abs().rolling(20, min_periods=15).sum())
# 2) efficiency_ratio_60: Kaufman efficiency ratio, 60d
f_er60 = per_asset(panel, lambda s: (s - s.shift(60)).abs() / s.diff().abs().rolling(60, min_periods=40).sum())
# 3) trend_accel_20_60: 20d momentum minus 60d momentum (slope acceleration)
f_accel = per_asset(panel, lambda s: (s / s.shift(20) - 1.0) - (s / s.shift(60) - 1.0))
# 4) vol_ratio_20_60: short/medium realized-vol ratio (vol term structure)
f_volr = per_asset(panel, lambda s: s.pct_change().rolling(20, min_periods=15).std()
                   / s.pct_change().rolling(60, min_periods=40).std())
# 5) autocorr_5_60: 5-lag return autocorrelation measured on 60d window (mean-reversion signature)
def _ac5(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return np.nan
    r = np.diff(x) / x[:-1]
    r = r[~np.isnan(r)]
    if len(r) < 25:
        return np.nan
    m = r - r.mean()
    denom = (m * m).sum()
    if denom <= 0:
        return np.nan
    return float((m[5:] * m[:-5]).sum() / denom)
f_ac5 = per_asset(panel, lambda s: s.rolling(60, min_periods=40).apply(_ac5, raw=True))
# 6) kurt_60: rolling 60d excess kurtosis of returns (tail risk)
f_kurt = per_asset(panel, lambda s: s.pct_change().rolling(60, min_periods=40).kurt())
# 7) gap_range_ratio_20: mean |overnight gap| / mean intraday range, 20d
def _build_gap_ratio(w=20, minp=12):
    out = {}
    for a in panel.columns:
        c = panel[a].dropna()
        op = open_p[a].reindex(c.index).dropna()
        hi = high_p[a].reindex(c.index).dropna()
        lo = low_p[a].reindex(c.index).dropna()
        idx = c.index.intersection(op.index).intersection(hi.index).intersection(lo.index)
        cc, oo, hh, ll = c[idx], op[idx], hi[idx], lo[idx]
        prev = cc.shift(1)
        gap = (oo / prev - 1.0).abs()
        rng = (hh - ll) / prev
        ratio = gap.rolling(w, min_periods=minp).mean() / rng.rolling(w, min_periods=minp).mean()
        out[a] = ratio.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)
f_gap = _build_gap_ratio()

cands = {
    "eff_ratio_20": f_er20,
    "eff_ratio_60": f_er60,
    "trend_accel_20_60": f_accel,
    "vol_ratio_20_60": f_volr,
    "autocorr_5_60": f_ac5,
    "kurt_60": f_kurt,
    "gap_range_ratio_20": f_gap,
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    p = report(name, m)
    results[name] = {"metrics": m, "pass": p}
    print(f"    decay: {m['decay_ic_by_horizon']}")

print("\n=== REGIME BREAKDOWN (10d IC by sub-period) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd_cache[str(ADM_H)]).dropna()
    parts = []
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            parts.append(f"{r0[:4]}:ic={sub.mean():+.4f}/icir={(sub.mean()/sd if sd>0 else 0):+.3f}/n={len(sub)}")
    print(f"  {name:22s} | " + " | ".join(parts))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner3_cycle28_explore_results.json", "w"), indent=1, default=float)
print("\nDONE cycle28 exploration")
