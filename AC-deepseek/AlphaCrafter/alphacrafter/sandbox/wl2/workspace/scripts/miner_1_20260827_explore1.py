"""miner_1 2026-08-27 exploration batch 1.

Candidates (all distinct from existing library):
  vwap_dev_20      : close vs 20d volume-weighted typical price deviation (vol-price pressure)
  updown_vol_20    : accumulation ratio = up-day volume / (up+down volume), 20d
  residual_mom_60  : SPX-residual cumulative return over 60d (idiosyncratic momentum)
  upper_shadow_20  : mean(upper_shadow - lower_shadow) over 20d (intraday rejection asymmetry)
  money_flow_20    : Chaikin money flow over 20d (volume-weighted range position)
  range_ratio_10x60: short/long range expansion (mean((h-l)/c,10) / mean((h-l)/c,60) - 1)

IC = daily cross-sectional Spearman(factor_t, fwd10_t) on own-calendar returns.
Gates: |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import json, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import miner_1_20260813_lib as L

ASSETS = L.ASSETS
GRID = L.GRID
N_GRID = L.N_GRID

series = L.asset_series()  # sym -> df[close, ret, fwd10] on own calendar
print(f"assets loaded: {len(series)}")

# ---------------------------------------------------------------- helpers
def grid_from_dict(d):
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s in d:
            ser = d[s]
            mat[:, j] = ser.reindex(GRID).values
    return mat


def add_factor(name, fdict):
    mat = grid_from_dict(fdict)
    ics = L.spearman_ic_matrix(mat, fwd10_mat)
    summ = L.summarize(ics, GRID, name, L.HORIZON)
    decay = L.decay_curve(mat, fwd_by_h)
    rank_mat = L.cross_sectional_rank(mat)
    to = L.turnover_10d_rank(rank_mat)
    cov_ad, cov_d8 = L.coverage_stats(mat)
    lpc, lname, lmax = L.library_pairwise_corr(mat)
    return {"metrics": {
        "ic": round(summ["ic"], 4), "icir": round(summ["icir"], 4),
        "ic_hit_ratio": round(summ["hit"], 4), "n_ic_dates": summ["n_ic_dates"],
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10_rank": round(to, 4), "decay_ic_by_horizon": decay,
        "max_abs_library_correlation": round(lmax, 4),
        "library_pairwise_corr": lpc,
        "regime": summ["regime"]},
        "pass": abs(summ["ic"]) >= 0.0070 and abs(summ["icir"]) >= 0.0840}


fwd10_mat = grid_from_dict({s: d["fwd10"] for s, d in series.items()})
fwd_by_h = L.fwd_by_horizon_dict(series)

results = {}

# ------------------------------------------------- 1. vwap_dev_20
print("\n=== vwap_dev_20 ===")
fd = {}
for s, df in series.items():
    h, l, c, v = df["close"], df["close"], df["close"], df["close"]
    # need OHLCV: rebuild from load_asset
    fd[s] = pd.Series(np.nan, index=df.index)
series2 = L.asset_series()
raw = {}
for s in ASSETS:
    d = L.load_asset(s)
    if d is None or len(d) < 100:
        continue
    typ = (d["high"] + d["low"] + d["close"]) / 3.0
    pv = (typ * d["volume"]).rolling(20).sum()
    vv = d["volume"].rolling(20).sum()
    raw[s] = (d["close"] / (pv / vv) - 1.0)
results["vwap_dev_20"] = add_factor("vwap_dev_20", raw)
print(json.dumps({k: v for k, v in results["vwap_dev_20"]["metrics"].items()
                  if k not in ("library_pairwise_corr", "regime", "decay_ic_by_horizon")}, indent=1))

# ------------------------------------------------- 2. updown_vol_20
print("\n=== updown_vol_20 ===")
raw = {}
for s in ASSETS:
    d = L.load_asset(s)
    if d is None or len(d) < 100:
        continue
    ret = d["close"].pct_change()
    up = d["volume"].where(ret > 0, 0.0).rolling(20).sum()
    dn = d["volume"].where(ret < 0, 0.0).rolling(20).sum()
    raw[s] = up / (up + dn) - 0.5
results["updown_vol_20"] = add_factor("updown_vol_20", raw)
print(json.dumps({k: v for k, v in results["updown_vol_20"]["metrics"].items()
                  if k not in ("library_pairwise_corr", "regime", "decay_ic_by_horizon")}, indent=1))

# ------------------------------------------------- 3. residual_mom_60
print("\n=== residual_mom_60 ===")
raw = {}
spx = L.load_asset("SPX")
spx_ret = spx["close"].pct_change()
for s in ASSETS:
    d = L.load_asset(s)
    if d is None or len(d) < 100:
        continue
    ret = d["close"].pct_change()
    mkt = spx_ret.reindex(d.index)
    W = 60
    out = pd.Series(np.nan, index=d.index)
    rv = ret.values
    mv = mkt.values
    for t in range(W, len(rv)):
        y = rv[t - W + 1:t + 1]
        x = mv[t - W + 1:t + 1]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < W * 0.7:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        resid = y[ok] - (a + b * x[ok])
        out.iloc[t] = resid.sum()
    raw[s] = out
results["residual_mom_60"] = add_factor("residual_mom_60", raw)
print(json.dumps({k: v for k, v in results["residual_mom_60"]["metrics"].items()
                  if k not in ("library_pairwise_corr", "regime", "decay_ic_by_horizon")}, indent=1))

# ------------------------------------------------- 4. upper_shadow_20
print("\n=== upper_shadow_20 ===")
raw = {}
for s in ASSETS:
    d = L.load_asset(s)
    if d is None or len(d) < 100:
        continue
    hi, lo, op, cl = d["high"], d["low"], d["open"], d["close"]
    rng = (hi - lo).replace(0, np.nan)
    us = (hi - np.maximum(op, cl)) / rng
    ls = (np.minimum(op, cl) - lo) / rng
    raw[s] = (us - ls).rolling(20).mean()
results["upper_shadow_20"] = add_factor("upper_shadow_20", raw)
print(json.dumps({k: v for k, v in results["upper_shadow_20"]["metrics"].items()
                  if k not in ("library_pairwise_corr", "regime", "decay_ic_by_horizon")}, indent=1))

# ------------------------------------------------- 5. money_flow_20 (Chaikin)
print("\n=== money_flow_20 ===")
raw = {}
for s in ASSETS:
    d = L.load_asset(s)
    if d is None or len(d) < 100:
        continue
    hi, lo, cl, v = d["high"], d["low"], d["close"], d["volume"]
    rng = (hi - lo).replace(0, np.nan)
    mfm = ((cl - lo) - (hi - cl)) / rng
    mfv = mfm * v
    raw[s] = mfv.rolling(20).sum() / v.rolling(20).sum()
results["money_flow_20"] = add_factor("money_flow_20", raw)
print(json.dumps({k: v for k, v in results["money_flow_20"]["metrics"].items()
                  if k not in ("library_pairwise_corr", "regime", "decay_ic_by_horizon")}, indent=1))

# ------------------------------------------------- 6. range_ratio_10x60
print("\n=== range_ratio_10x60 ===")
raw = {}
for s in ASSETS:
    d = L.load_asset(s)
    if d is None or len(d) < 100:
        continue
    rng = (d["high"] - d["low"]) / d["close"]
    r10 = rng.rolling(10).mean()
    r60 = rng.rolling(60).mean()
    raw[s] = r10 / r60 - 1.0
results["range_ratio_10x60"] = add_factor("range_ratio_10x60", raw)
print(json.dumps({k: v for k, v in results["range_ratio_10x60"]["metrics"].items()
                  if k not in ("library_pairwise_corr", "regime", "decay_ic_by_horizon")}, indent=1))

# ------------------------------------------------- summary
print("\n===== SUMMARY =====")
for k, v in results.items():
    m = v["metrics"]
    print(f"{k:18s} pass={v['pass']}  IC={m['ic']:+.4f}  ICIR={m['icir']:+.4f}  "
          f"hit={m['ic_hit_ratio']:.3f}  maxcorr={m['max_abs_library_correlation']:.3f}  "
          f"cov_d8={m['coverage_dates_ge8']:.3f}")

with open("scripts/_miner1_20260827_explore1_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/_miner1_20260827_explore1_results.json")
