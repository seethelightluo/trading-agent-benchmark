"""Exploration 2026-11-19 (miner_2): drawdown / recovery-quality factor family.

Idea: The ensemble already captures raw momentum (mom_10d, mom_120d) and
volatility (vol_of_vol). A complementary dimension is the *quality* of the
trend path - how much underwater risk / drawdown each asset has been through
relative to its trend, and how far price sits below its peak. Assets that hold
near highs with shallow drawdowns differ cross-sectionally from ones that are
rebounding from deep drawdowns even at the same momentum level.

Variants (horizon N in 10/21/63):
  - dd_depth_Nd : (close - rolling_min(close,N)) / rolling_max(close,N)  [normalized drawdown depth, higher=shallower=better]
  - dd_from_peak_Nd : close / rolling_max(close,N)                        [how close to recent peak]
  - recovery_Nd : close / rolling_max(close,N) * (cumRet sign quality)    [peak proximity scaled by directional bias]

Mixed calendars handled per-asset before reindexing (per helper convention).
Admission gate (15-asset universe): abs daily paper IC >= 0.0070 AND
abs daily paper ICIR >= 0.0840 at horizon h=10, min_valid=8.
Window: 2020-01-01 .. 2026-11-18 (visible through previous completed trading day).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260730_helpers import (WATCH, forward_returns,
                                      factor_ic_report, factor_turnover,
                                      coverage, decay_report,
                                      max_library_correlation)

MAXD = "2026-11-18"
MIND = "2020-01-01"

def load_close(symbol, root="../persistent"):
    for sub, name in (("stock_data", symbol), ("index_data", symbol)):
        p = f"{root}/{sub}/{name}.csv"
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            continue
        df = df[(df["date"] >= MIND) & (df["date"] <= MAXD)].set_index("date")
        return df
    return None

closes = {a: load_close(a)["close"].astype(float) for a in WATCH}
panel = pd.DataFrame(closes).dropna(how="all").sort_index()
rets = panel.pct_change()
print(f"closes shape: {panel.shape}  range: {panel.index.min().date()} -> {panel.index.max().date()}")

def per_asset_series(fn, N, series=None):
    src = rets if series is None else series
    cols = {}
    for a in WATCH:
        s = src[a].dropna()
        cols[a] = fn(s, N).reindex(panel.index)
    return pd.DataFrame(cols)

def dd_depth(c, N):
    roll_max = c.rolling(N).max()
    roll_min = c.rolling(N).min()
    # higher = price nearer max, shallower drawdown
    return (c - roll_min) / (roll_max - roll_min + 1e-12)

def close_to_peak(c, N):
    roll_max = c.rolling(N).max()
    return c / roll_max

def recovery_bias(c, N):
    # peak proximity combined with positive return bias (trend going up from the high)
    roll_max = c.rolling(N).max()
    prox = c / roll_max
    pos = c.diff().clip(lower=0).rolling(N).mean()
    neg = (-c.diff()).clip(lower=0).rolling(N).mean()
    bias = (pos / (neg + 1e-12)) - 1.0
    return prox * (1.0 + bias)

def run_var(name, fpanel):
    h = 10
    fwd = forward_returns(rets, h)
    rep = factor_ic_report(fpanel, fwd, horizon=h)
    if rep is None:
        print(f"{name:<26} insufficient data"); return None
    turn = factor_turnover(fpanel)
    cov = coverage(fpanel)
    dec = decay_report(fpanel, rets)
    rho, rho_map = max_library_correlation(fpanel)
    passed = abs(rep["ic"]) >= 0.0070 and abs(rep["icir"]) >= 0.0840
    print(f"{name:<26} IC={rep['ic']:>8.4f} ICIR={rep['icir']:>8.4f} "
          f"hit={rep['ic_hit_ratio']:>5.3f} n={rep['n_ic_dates']:>5d} "
          f"meanN={rep['mean_n_valid']:>4.1f} turn={turn:>5.2f} "
          f"cov_date8={cov['coverage_dates_ge8']:>4.2f} decay1/5/10/20="
          f"{dec['1']}/{dec['5']}/{dec['10']}/{dec['20']} "
          f"maxRho={rho:>5.3f} PASS={passed}")
    return dict(rep=rep, turnover=turn, coverage=cov, rho=rho, panel=fpanel, name=name, passed=passed)

# needs close series for drawdown-based factors
close_panel = panel

print("\n== Drawdown / recovery-quality factor family ==\n")
res = []
for N in (10, 21, 63):
    res.append(run_var(f"dd_depth_{N}d", per_asset_series(dd_depth, N, close_panel)))
    res.append(run_var(f"close_to_peak_{N}d", per_asset_series(close_to_peak, N, close_panel)))
    res.append(run_var(f"recovery_bias_{N}d", per_asset_series(recovery_bias, N, close_panel)))

import pickle
best = [r for r in res if r and r["passed"]]
print(f"\nPassed candidates: {len(best)}")
for r in best:
    print(" ", r["name"], "IC", round(r["rep"]["ic"],4), "ICIR", round(r["rep"]["icir"],4), "rho", round(r["rho"],3))
    with open(f"scripts/_miner2_candidate_{r['name']}.pkl","wb") as f:
        pickle.dump(r, f)