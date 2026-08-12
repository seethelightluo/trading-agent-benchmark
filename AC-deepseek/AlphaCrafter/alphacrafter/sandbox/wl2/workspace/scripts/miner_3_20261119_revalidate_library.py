"""miner_3 2026-11-19 re-validation of all EFFECTIVE library factors on fresh data
visible through 2026-11-18 (current sim date 2026-11-19).

Recomputes every factor from raw OHLCV (own-calendar per asset), reindexes to the
master grid, and reports 10d-forward rank IC / ICIR, regime breakdown, coverage,
turnover, and gate pass/fail (|IC|>=0.0070, |ICIR|>=0.0840). Also recomputes the
macro-input factors (DXY/USDJPY/VIX conditional betas) using observation-only data.
"""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, coverage_stats, safe_div,
                                  load_macro, MIN_ASSETS)

GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["rng_pct"] = (df["high"] - df["low"]) / df["close"]
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["hl_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)

spx_ret = series["SPX"]["ret"] if "SPX" in series else None
macro = {m: load_macro(m) for m in ["DXY", "USDJPY", "VIX", "USDCNY", "EURUSD"]}
macro = {m: s for m, s in macro.items() if s is not None}
macro_ret = {m: s.pct_change() for m, s in macro.items()}
print("macro loaded:", sorted(macro.keys()))


def roll_beta(a, b, w, minp=30, cond=None):
    """rolling beta of a on b (own calendar). cond: mask for b (e.g. b<0)."""
    df = pd.concat([a, b], axis=1, join="outer")
    df.columns = ["a", "b"]
    if cond is not None:
        mask = cond.reindex(df.index)
        df = df.where(mask, np.nan)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["b"])
    var = df["b"].rolling(w, min_periods=minp).var()
    return (cov / var).reindex(a.index)


def rmean(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()


def rstd(s, w, minp):
    return s.rolling(w, min_periods=minp).std()


def longest_run(pos_mask):
    """rolling 20d longest consecutive True run in boolean series."""
    arr = pos_mask.astype(float).values
    out = np.full(len(arr), np.nan)
    run = 0
    for i in range(len(arr)):
        run = run + 1 if arr[i] == 1 else 0
        out[i] = run
    return pd.Series(out, index=pos_mask.index).rolling(20, min_periods=10).max()


def report(name, cand, exp_dir=1):
    mat = to_grid(cand)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} GATE={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    return {"ic": ic, "icir": icir, "q": q, "ok": ok, "hit": summ["hit"],
            "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"], "decay": dec,
            "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_d8,
            "turnover_10d_rank": to}


results = {}

# ---- 1. max_consec_gain_20 ----
cand = {}
for s, df in series.items():
    pos = df["ret"] > 0
    cand[s] = longest_run(pos)
results["max_consec_gain_20"] = report("max_consec_gain_20", cand)

# ---- 2. max_consec_loss_20 ----
cand = {}
for s, df in series.items():
    neg = df["ret"] < 0
    cand[s] = longest_run(neg)
results["max_consec_loss_20"] = report("max_consec_loss_20", cand)

# ---- 3. mom20_volproxy60 ----
cand = {}
for s, df in series.items():
    mom20 = df["close"].shift(5) / df["close"].shift(25) - 1.0
    mom60p = (df["close"].shift(5) / df["close"].shift(65) - 1.0).abs()
    cand[s] = mom20 / (1.0 + mom60p)
results["mom20_volproxy60"] = report("mom20_volproxy60", cand)

# ---- 4. spx_corr60 ----
cand = {}
for s, df in series.items():
    if s == "SPX":
        cand[s] = pd.Series(1.0, index=df.index)
    else:
        j = pd.concat([df["ret"], spx_ret], axis=1, join="outer")
        j.columns = ["a", "b"]
        cand[s] = j["a"].rolling(60, min_periods=15).corr(j["b"]).reindex(df.index)
results["spx_corr60"] = report("spx_corr60", cand)

# ---- 5. mom_20d_skip5 ----
cand = {s: (df["close"].shift(5) / df["close"].shift(25) - 1.0) for s, df in series.items()}
results["mom_20d_skip5"] = report("mom_20d_skip5", cand)

# ---- 6. gain_loss_20 ----
cand = {}
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0).rolling(20, min_periods=10).mean()
    dn = r.clip(upper=0).rolling(20, min_periods=10).mean().abs()
    cand[s] = up / (dn + 1e-9)
results["gain_loss_20"] = report("gain_loss_20", cand)

# ---- 7. downbeta_spx_60 ----
cand = {}
for s, df in series.items():
    if s == "SPX":
        cand[s] = pd.Series(1.0, index=df.index)
    else:
        cand[s] = roll_beta(df["ret"], spx_ret, 60, minp=30, cond=spx_ret < 0)
results["downbeta_spx_60"] = report("downbeta_spx_60", cand)

# ---- 8. usdjpy_beta_cond_120x60 ----
cand = {}
usdjpy_r = macro_ret.get("USDJPY")
if usdjpy_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], usdjpy_r, 120, minp=60)
        mom = usdjpy_r.rolling(60, min_periods=30).mean()
        cand[s] = (beta * mom).reindex(df.index)
results["usdjpy_beta_cond_120x60"] = report("usdjpy_beta_cond_120x60", cand)

# ---- 9. volcluster_60 ----
cand = {}
for s, df in series.items():
    rv = rstd(df["ret"], 20, 5)
    cand[s] = rv.rolling(60, min_periods=15).std()
results["volcluster_60"] = report("volcluster_60", cand)

# ---- 10. calmness_20 ----
cand = {}
for s, df in series.items():
    sd = rstd(df["ret"], 20, 10)
    calm = (df["ret"].abs() < 0.5 * sd).astype(float)
    cand[s] = rmean(calm, 20, 10)
results["calmness_20"] = report("calmness_20", cand)

# ---- 11. close_pos_20 ----
cand = {s: rmean(df["hl_pos"], 20, 10) for s, df in series.items()}
results["close_pos_20"] = report("close_pos_20", cand)

# ---- 12. days_since_high_60 ----
cand = {}
for s, df in series.items():
    rollmax = df["close"].rolling(60, min_periods=40).max()
    is_high = (df["close"] >= rollmax).astype(float)
    days = np.nan
    out = []
    v = np.full(len(df), np.nan)
    last_high = -1
    for i in range(len(df)):
        if is_high.iloc[i] == 1:
            last_high = i
        if last_high >= 0 and i - 59 >= 0:
            v[i] = i - last_high
    cand[s] = pd.Series(v, index=df.index)
results["days_since_high_60"] = report("days_since_high_60", cand)

# ---- 13. lagbeta_spx_60 ----
cand = {}
for s, df in series.items():
    if s == "SPX":
        cand[s] = pd.Series(1.0, index=df.index)
    else:
        lag = spx_ret.shift(1)
        cand[s] = roll_beta(df["ret"], lag, 60, minp=30)
results["lagbeta_spx_60"] = report("lagbeta_spx_60", cand)

# ---- 14. intraday_drift_20 ----
cand = {s: rmean(df["intraday"], 20, 10) for s, df in series.items()}
results["intraday_drift_20"] = report("intraday_drift_20", cand)

# ---- 15. dxy_beta_cond_60x20 ----
cand = {}
dxy_r = macro_ret.get("DXY")
if dxy_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], dxy_r, 60, minp=30)
        mom = dxy_r.rolling(20, min_periods=10).mean()
        cand[s] = (beta * mom).reindex(df.index)
results["dxy_beta_cond_60x20"] = report("dxy_beta_cond_60x20", cand)

# ---- 16. vix_beta_cond_60x20 ----
cand = {}
vix_r = macro_ret.get("VIX")
if vix_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], vix_r, 60, minp=30)
        mom = vix_r.rolling(20, min_periods=10).mean()
        cand[s] = (-beta * mom).reindex(df.index)
results["vix_beta_cond_60x20"] = report("vix_beta_cond_60x20", cand)

# ---- 17. mom_10d_skip5 ----
cand = {s: (df["close"].shift(5) / df["close"].shift(15) - 1.0) for s, df in series.items()}
results["mom_10d_skip5"] = report("mom_10d_skip5", cand)

# ---- 18. mom_120d_skip5 ----
cand = {s: (df["close"].shift(5) / df["close"].shift(125) - 1.0) for s, df in series.items()}
results["mom_120d_skip5"] = report("mom_120d_skip5", cand)

# ---- 19. mom_180d_skip5 ----
cand = {s: (df["close"].shift(5) / df["close"].shift(185) - 1.0) for s, df in series.items()}
results["mom_180d_skip5"] = report("mom_180d_skip5", cand)

# ---- 20. mom30_vol60 ----
cand = {}
for s, df in series.items():
    mom30 = df["close"].shift(5) / df["close"].shift(35) - 1.0
    vol60 = rstd(df["ret"], 60, 15)
    cand[s] = mom30 / vol60
results["mom30_vol60"] = report("mom30_vol60", cand)

# ---- 21. range_pos_252 ----
cand = {}
for s, df in series.items():
    lo = df["close"].rolling(252, min_periods=30).min()
    hi = df["close"].rolling(252, min_periods=30).max()
    cand[s] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)
results["range_pos_252"] = report("range_pos_252", cand)

# ---- 22. vol_of_vol20x60 ----
cand = {}
for s, df in series.items():
    rv20 = rstd(df["ret"], 20, 5)
    cand[s] = rv20.rolling(60, min_periods=15).std()
results["vol_of_vol20x60"] = report("vol_of_vol20x60", cand)

json.dump(results, open("scripts/miner_3_20261119_revalidate_results.json", "w"), indent=1, default=str)
print("\nDONE - summary of gate status:")
for k, v in results.items():
    if v is not None:
        print(f"  {k:28s} IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} GATE={v['ok']}")
