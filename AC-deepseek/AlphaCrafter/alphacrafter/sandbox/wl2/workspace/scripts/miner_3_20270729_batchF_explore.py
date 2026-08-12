"""miner_3 2027-07-29 batch F: novel factor exploration (data through 2027-07-28).

New families not covered by the 36 existing signal artifacts:
  rel_mom20 / rel_mom10      breadth-relative momentum (minus cross-sectional median)
  rev5_vol20 / rev10_vol20   vol-scaled short-term reversal
  rng_eff20                  range efficiency (intraday range vs close-to-close vol)
  gap_ma10                   overnight-gap mean reversion
  us10y_beta_cond_60x20      rate beta conditioned on 10Y momentum
  autocorr_10                return autocorrelation (trend persistence)
  maxdd_60                   drawdown depth from running max
  skew60 / kurt20            higher moments
  mom20_trendfilt            momentum gated by 60d trend sign
  vol_ratio_10_60            vol term-structure ratio
  btc_beta_60                risk-on beta vs BTC composite

Gates: |IC|>=0.0070, |ICIR|>=0.0840 on 10d rank IC. Uses own-calendar per asset.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  summarize, safe_div, load_macro, MIN_ASSETS,
                                  cross_sectional_rank, library_pairwise_corr,
                                  turnover_10d_rank)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
OUT = "scripts/miner_3_20270729_batchF_results.json"

print(f"grid rows: {len(GRID)} (2020-01-01 .. {GRID[-1]})", flush=True)


def load_asset(sym, days=2200):
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
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}", flush=True)
print(f"data end: {max(df.index[-1] for df in series.values())}", flush=True)

spx_ret = series["SPX"]["ret"] if "SPX" in series else None
macro = {m: load_macro(m) for m in ["DXY", "USDJPY", "VIX", "USDCNY", "EURUSD"]}
macro = {m: s for m, s in macro.items() if s is not None}
macro_ret = {m: s.pct_change() for m, s in macro.items()}
print("macro loaded:", sorted(macro.keys()), flush=True)


def cs_rank(mat):
    T, n = mat.shape
    out = np.full_like(mat, np.nan, dtype=float)
    for t in range(T):
        row = mat[t]
        valid = ~np.isnan(row)
        if valid.sum() < MIN_ASSETS:
            continue
        vals = row[valid]
        ranks = pd.Series(vals).rank(pct=True).values
        out[t, valid] = ranks
    return out


def rank_ic(factor_rank, fwd_rank):
    T = factor_rank.shape[0]
    ics = []
    for t in range(T):
        f = factor_rank[t]
        r = fwd_rank[t]
        ok = ~(np.isnan(f) | np.isnan(r))
        if ok.sum() < MIN_ASSETS:
            continue
        v = pd.Series(f[ok]).rank().corr(pd.Series(r[ok]).rank())
        if np.isfinite(v):
            ics.append((t, float(v)))
    return ics


def fwd_rank_matrices(series_dict, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        d = {}
        for s, df in series_dict.items():
            close = df["close"]
            d[s] = close.shift(-h) / close - 1.0
        out[h] = cs_rank(to_grid(d))
    return out


fwd_ranks = fwd_rank_matrices(series)


def roll_beta(a, b, w, minp=30, cond=None):
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


def max_drawdown_depth(close, w):
    rollmax = close.rolling(w, min_periods=40).max()
    dd = close / rollmax - 1.0
    return dd.rolling(w, min_periods=40).min()


def report(name, cand, fwd_ranks):
    mat = to_grid(cand)
    rank_mat = cs_rank(mat)
    ics = rank_ic(rank_mat, fwd_ranks[10])
    dates = np.array(GRID)
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES", flush=True)
        return None
    valid = ~np.isnan(mat)
    cov_ad = float(valid.mean())
    cov_d8 = float(np.mean(valid.sum(axis=1) >= MIN_ASSETS))
    to = turnover_10d_rank(rank_mat)
    dec = {}
    for h, fr in fwd_ranks.items():
        icl = rank_ic(rank_mat, fr)
        if icl:
            dec[str(h)] = round(float(np.mean([v for _, v in icl])), 4)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    pc, pc_name, pc_max = library_pairwise_corr(mat)
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} "
          f"maxlibcorr={pc_max:.3f}({pc_name}) GATE={ok}", flush=True)
    print("   regime:", {k: v for k, v in summ["regime"].items()}, flush=True)
    print("   decay:", dec, flush=True)
    return {"ic": ic, "icir": icir, "q": q, "ok": ok, "hit": summ["hit"],
            "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"], "decay": dec,
            "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_d8,
            "turnover_10d_rank": to, "max_abs_library_correlation": pc_max,
            "max_corr_factor": pc_name}


results = {}
def run(name, cand):
    if name in results and results[name] is not None:
        print(f"skip {name}", flush=True)
        return results[name]
    results[name] = report(name, cand, fwd_ranks)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"  [saved {len(results)}]", flush=True)
    return results[name]


# 1 rel_mom20: 20d momentum (skip5) minus cross-sectional median
mom20 = {}
for s, df in series.items():
    mom20[s] = df["close"].shift(5) / df["close"].shift(25) - 1.0
mom20_mat = to_grid(mom20)
mom20_med = np.nanmedian(mom20_mat, axis=1, keepdims=True)
rel_mat = mom20_mat - mom20_med
rel_ser = {}
for j, s in enumerate(ASSETS):
    if s in series:
        rel_ser[s] = pd.Series(rel_mat[:, j], index=GRID)
run("rel_mom20", rel_ser)

# 2 rel_mom10: 10d momentum (skip1) minus cross-sectional median
mom10 = {}
for s, df in series.items():
    mom10[s] = df["close"].shift(1) / df["close"].shift(11) - 1.0
mom10_mat = to_grid(mom10)
mom10_med = np.nanmedian(mom10_mat, axis=1, keepdims=True)
rel10_mat = mom10_mat - mom10_med
rel10_ser = {}
for j, s in enumerate(ASSETS):
    if s in series:
        rel10_ser[s] = pd.Series(rel10_mat[:, j], index=GRID)
run("rel_mom10", rel10_ser)

# 3 rev5_vol20: -5d return / vol20 (short-term reversal scaled)
rev5v = {}
for s, df in series.items():
    r5 = df["close"].shift(1) / df["close"].shift(6) - 1.0
    v20 = rstd(df["ret"], 20, 10)
    rev5v[s] = pd.Series(safe_div(-r5, v20), index=df.index)
run("rev5_vol20", rev5v)

# 4 rev10_vol20
rev10v = {}
for s, df in series.items():
    r10 = df["close"].shift(1) / df["close"].shift(11) - 1.0
    v20 = rstd(df["ret"], 20, 10)
    rev10v[s] = pd.Series(safe_div(-r10, v20), index=df.index)
run("rev10_vol20", rev10v)

# 5 rng_eff20: mean range / close-to-close vol
rng_eff = {}
for s, df in series.items():
    mr = rmean(df["rng_pct"], 20, 10)
    v = rstd(df["ret"], 20, 10)
    rng_eff[s] = pd.Series(safe_div(mr, v), index=df.index)
run("rng_eff20", rng_eff)

# 6 gap_ma10: mean overnight gap (reversal candidate)
gap_ma = {}
for s, df in series.items():
    gap_ma[s] = rmean(df["gap"], 10, 5)
run("gap_ma10", gap_ma)

# 7 us10y_beta_cond_60x20: beta to US10Y daily change conditioned on 10Y 20d momentum
us10y_r = series["US10Y"]["ret"] if "US10Y" in series else None
ub = {}
if us10y_r is not None:
    for s, df in series.items():
        if s == "US10Y":
            ub[s] = pd.Series(1.0, index=df.index)
        else:
            beta = roll_beta(df["ret"], us10y_r, 60, minp=30)
            m10 = us10y_r.rolling(20, min_periods=10).mean()
            ub[s] = (beta * m10).reindex(df.index)
    run("us10y_beta_cond_60x20", ub)

# 8 autocorr_10: 10d return autocorrelation
ac10 = {}
for s, df in series.items():
    r = df["ret"]
    ac10[s] = r.rolling(10, min_periods=6).apply(
        lambda x: pd.Series(x).autocorr() if len(x) >= 6 else np.nan, raw=False)
run("autocorr_10", ac10)

# 9 maxdd_60: max drawdown depth over 60d (negative; more negative = deeper)
mdd = {}
for s, df in series.items():
    mdd[s] = max_drawdown_depth(df["close"], 60)
run("maxdd_60", mdd)

# 10 skew60: 60d return skewness
sk60 = {}
for s, df in series.items():
    sk60[s] = df["ret"].rolling(60, min_periods=30).skew()
run("skew60", sk60)

# 11 kurt20: 20d kurtosis
ku20 = {}
for s, df in series.items():
    ku20[s] = df["ret"].rolling(20, min_periods=12).kurt()
run("kurt20", ku20)

# 12 mom20_trendfilt: 20d momentum gated by sign(60d momentum)
mtf = {}
for s, df in series.items():
    m20 = df["close"].shift(5) / df["close"].shift(25) - 1.0
    m60 = df["close"].shift(5) / df["close"].shift(65) - 1.0
    mtf[s] = m20 * np.sign(m60)
run("mom20_trendfilt", mtf)

# 13 vol_ratio_10_60: vol10 / vol60
vr = {}
for s, df in series.items():
    v10 = rstd(df["ret"], 10, 5)
    v60 = rstd(df["ret"], 60, 15)
    vr[s] = pd.Series(safe_div(v10, v60), index=df.index)
run("vol_ratio_10_60", vr)

# 14 btc_beta_60: plain 60d beta vs BTC returns
btc_r = series["BTC"]["ret"] if "BTC" in series else None
bb = {}
if btc_r is not None:
    for s, df in series.items():
        if s == "BTC":
            bb[s] = pd.Series(1.0, index=df.index)
        else:
            bb[s] = roll_beta(df["ret"], btc_r, 60, minp=30)
    run("btc_beta_60", bb)

print("\nDONE. saved", OUT)
