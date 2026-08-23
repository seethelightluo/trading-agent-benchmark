"""Exploration 2026-10-22 (miner_2): return-bias / trend-quality factor family.

Idea: Beyond raw momentum (already in ensemble: mom_10d, mom_120d) and vol
factors, the *quality* of the trend - whether up-moves are larger/faster than
down-moves and whether the path is persistent - may carry incremental
cross-sectional signal. This family measures the asymmetry of the return
distribution over a horizon.

Variants (each horizon N in 10/21/63):
  - bias_ratio_Nd : mean(+ret)/|mean(-ret)| over N days (return asymmetry)
  - win_rate_Nd   : fraction of days with positive return over N days
  - tq_combine_Nd : sign-consistent combo = (win_rate-0.5) * (sum|ret|/std)

Mixed calendars handled per-asset before reindexing.
Admission gate (15-asset universe): abs daily paper IC >= 0.0070 AND
abs daily paper ICIR >= 0.0840 at horizon h=10, min_valid=8.
Window: 2020-01-01 .. 2026-10-21 (visible through previous completed trading day).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260730_helpers import (WATCH, load_panel, forward_returns,
                                      factor_ic_report, factor_turnover,
                                      coverage, decay_report,
                                      max_library_correlation)

MAXD = "2026-10-21"
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

def per_asset_series(fn, N):
    cols = {}
    for a in WATCH:
        r = rets[a].dropna()
        cols[a] = fn(r, N).reindex(panel.index)
    return pd.DataFrame(cols)

def bias_ratio(r, N):
    pos = r[r > 0]
    neg = r[r < 0]
    pr = pos.rolling(N).mean()
    nr = neg.rolling(N).mean().abs()
    return (pr / nr).replace([np.inf, -np.inf], np.nan)

def win_rate(r, N):
    return (r > 0).rolling(N).mean()

def tq_combine(r, N):
    # sign-consistent persistence: demeaned win rate x trending magnitude
    wr = (r > 0).rolling(N).mean()
    mag = r.abs().rolling(N).sum()
    return (wr - 0.5) * mag

def run_var(name, fpanel):
    h = 10
    fwd = forward_returns(rets, h)
    rep = factor_ic_report(fpanel, fwd, horizon=h)
    if rep is None:
        print(f"{name:<24} insufficient data"); return None
    turn = factor_turnover(fpanel)
    cov = coverage(fpanel)
    dec = decay_report(fpanel, rets)
    rho, _ = max_library_correlation(fpanel)
    passed = abs(rep["ic"]) >= 0.0070 and abs(rep["icir"]) >= 0.0840
    print(f"{name:<24} IC={rep['ic']:>8.4f} ICIR={rep['icir']:>8.4f} "
          f"hit={rep['ic_hit_ratio']:>5.3f} n={rep['n_ic_dates']:>5d} "
          f"meanN={rep['mean_n_valid']:>4.1f} turn={turn:>5.2f} "
          f"cov_date8={cov['coverage_dates_ge8']:>4.2f} decay1/5/10/20="
          f"{dec['1']}/{dec['5']}/{dec['10']}/{dec['20']} "
          f"maxRho={rho:>5.3f} PASS={passed}")
    return dict(rep=rep, turnover=turn, coverage=cov, rho=rho, panel=fpanel, name=name, passed=passed)

print("\n== Return-bias / trend-quality family ==\n")
res = []
for N in (10, 21, 63):
    res.append(run_var(f"bias_ratio_{N}d", per_asset_series(bias_ratio, N)))
    res.append(run_var(f"win_rate_{N}d", per_asset_series(win_rate, N)))
    res.append(run_var(f"tq_combine_{N}d", per_asset_series(tq_combine, N)))

# save best candidate signals for possible persistence
import pickle
best = [r for r in res if r and r["passed"]]
print(f"\nPassed candidates: {len(best)}")
for r in best:
    print(" ", r["name"], "IC", round(r["rep"]["ic"],4), "ICIR", round(r["rep"]["icir"],4), "rho", round(r["rho"],3))