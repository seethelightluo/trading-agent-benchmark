"""miner_1 2026-09-10 batch C: novel ORTHOGONAL factor exploration.

Goal: find factors that (a) pass |IC|>=0.007 & |ICIR|>=0.084 on fwd-10d cross-sectional
Spearman, and (b) are NOT redundant with the existing library (time-averaged |rho| < 0.5
vs every factors/*.signal.npy artifact).

Families tested (not already in library):
  - days_since_low_60   : days since 60d low (trend-reversal timing; symmetric to evicted days_since_high_60)
  - kurt60              : rolling excess kurtosis of daily returns (tail risk; library has skew only)
  - updown_vol_ratio_60 : upside vol / downside vol asymmetry (library has downbeta vs SPX, not own-asset)
  - beta_ndx_60         : 60d beta to NDX (tech-factor exposure; library has SPX/DXY/USDJPY/VIX betas)
  - corr_ret_vol_20     : rolling corr of daily ret with vol change (leverage-effect asymmetry)
  - range_ratio_20      : mean (high-low)/close over 20d (intraday range vol vs close-to-close)
  - vol_flow_5x60       : 5d avg volume / 60d avg volume (volume flow; expect partial coverage)
  - accel_mom_20x20     : momentum acceleration = (20d mom, skip5) - (20d mom, skip5, 20d ago)

Data visible through 2026-09-09.
"""
import sys, json, os, glob
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, N_GRID, HORIZON, to_grid, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, coverage_stats, load_asset)

series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is not None and len(df) > 120:
        close = df["close"].astype(float)
        ret = close.pct_change()
        d = pd.DataFrame({"close": close, "ret": ret, "volume": df["volume"].astype(float),
                          "high": df["high"].astype(float), "low": df["low"].astype(float)})
        d["logp"] = np.log(close)
        d["vol20"] = ret.rolling(20).std()
        d["vol60"] = ret.rolling(60).std()
        series[s] = d
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")

def rolling_corr(x, y, w):
    mx = x.rolling(w).mean(); my = y.rolling(w).mean()
    cov = ((x - mx) * (y - my)).rolling(w).sum()
    vx = ((x - mx) ** 2).rolling(w).sum()
    vy = ((y - my) ** 2).rolling(w).sum()
    return cov / np.sqrt(vx * vy)

def rolling_beta(x, y, w):
    mx = x.rolling(w).mean(); my = y.rolling(w).mean()
    cov = ((x - mx) * (y - my)).rolling(w).sum()
    vx = ((x - mx) ** 2).rolling(w).sum()
    return cov / vx

def rolling_kurt(x, w):
    mu = x.rolling(w).mean()
    sd = x.rolling(w).std(ddof=0)
    m4 = ((x - mu) ** 4).rolling(w).mean()
    return m4 / (sd ** 4) - 3.0

ndx_ret = series["NDX"]["ret"] if "NDX" in series else None

cands = {k: {} for k in ["days_since_low_60", "kurt60", "updown_vol_ratio_60", "beta_ndx_60",
                         "corr_ret_vol_20", "range_ratio_20", "vol_flow_5x60", "accel_mom_20x20"]}

for s, d in series.items():
    close, ret = d["close"], d["ret"]
    vol60 = d["vol60"]
    # days since 60d low (positive = recently made low)
    max_low = close.rolling(60, min_periods=30).min()
    dsl = pd.Series(np.nan, index=close.index)
    for i in range(60, len(close)):
        win = close.iloc[i-59:i+1]
        if win.min() == close.iloc[i]:
            dsl.iloc[i] = 0
        else:
            dsl.iloc[i] = dsl.iloc[i-1] + 1 if np.isfinite(dsl.iloc[i-1]) else np.nan
    cands["days_since_low_60"][s] = dsl
    # kurtosis 60d
    cands["kurt60"][s] = rolling_kurt(ret, 60)
    # upside/downside vol ratio
    up = ret.clip(lower=0); dn = ret.clip(upper=0).abs()
    v_up = up.rolling(60).std(); v_dn = dn.rolling(60).std()
    cands["updown_vol_ratio_60"][s] = v_up / (v_dn + 1e-12)
    # beta to NDX
    if ndx_ret is not None:
        nr = ndx_ret.reindex(GRID)
        r_al = ret.reindex(GRID)
        cands["beta_ndx_60"][s] = rolling_beta(nr, r_al, 60)
    # corr(ret, vol change) 20d
    dv = d["vol20"].diff()
    cands["corr_ret_vol_20"][s] = rolling_corr(ret, dv, 20)
    # range ratio 20d
    rng = (d["high"] - d["low"]) / close
    cands["range_ratio_20"][s] = rng.rolling(20).mean()
    # volume flow
    v = d["volume"]
    cands["vol_flow_5x60"][s] = v.rolling(5).mean() / (v.rolling(60).mean() + 1e-9)
    # momentum acceleration
    m = close / close.shift(20) - 1.0
    cands["accel_mom_20x20"][s] = m - m.shift(20)

# ---------- library correlation (time-averaged Spearman, proper) ----------
def library_corr_timeavg(factor_mat, min_dates=60):
    ours = cross_sectional_rank(factor_mat)
    out = {}
    for f in sorted(glob.glob("factors/*.signal.npy")):
        arr = np.load(f, allow_pickle=True)
        rows = min(arr.shape[0], ours.shape[0])
        a = ours[:rows]
        b = cross_sectional_rank(arr[:rows])
        rhos = []
        for t in range(rows):
            x = a[t]; y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 8:
                c = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
                if np.isfinite(c):
                    rhos.append(c)
        if len(rhos) >= min_dates:
            out[os.path.basename(f).replace(".signal.npy", "")] = round(float(np.mean(rhos)), 4)
    return out

# ---------- evaluate ----------
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}

def report(name, mat):
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(f"{name}: NO VALID IC DATES"); return None
    cov_ad, cov_d8 = coverage_stats(mat)
    turn = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    libc = library_corr_timeavg(mat)
    top = sorted(libc.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    ic, icir = summ["ic"], summ["icir"]
    gate_ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    top_rho = abs(top[0][1]) if top else 0.0
    print("=" * 110)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={turn:.3f} | GATE_IC={'PASS' if gate_ok else 'FAIL'} "
          f"maxlib_rho={top_rho:.3f}")
    print("  regime:", {k: f"{v['ic']:+.3f}/{v['icir']:+.2f}(n={v['n']})" for k, v in summ["regime"].items()})
    print("  decay:", dec)
    print("  top lib corr:", top)
    return {"ic": ic, "icir": icir, "hit": summ["hit"], "n": summ["n_ic_dates"],
            "cov_ad": cov_ad, "cov_d8": cov_d8, "turn": turn, "decay": dec,
            "regime": summ["regime"], "gate_ok": gate_ok, "maxlib_rho": top_rho,
            "top_libcorr": top}

for name, cd in cands.items():
    mat = to_grid(cd)
    results[name] = report(name, mat)

print("\n===== SUMMARY =====")
for k, v in results.items():
    if v is None:
        continue
    print(f"{k:22s} IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} n={v['n']} turn={v['turn']:.3f} "
          f"maxlib={v['maxlib_rho']:.3f} GATE={'PASS' if v['gate_ok'] and v['maxlib_rho'] < 0.5 else '--'}")

json.dump(results, open("scripts/miner_1_20260910_batchC_results.json", "w"), indent=1, default=str)
print("saved scripts/miner_1_20260910_batchC_results.json")
