"""miner_2 2027-02-25: explore NEW candidate factor families (not in library).

Data visible through 2027-02-24 (sim date 2027-02-25).
IC = daily cross-sectional Spearman of factor rank vs 10d fwd return rank.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840.

Candidates (all distinct from existing library factors):
  gap_mom_20       : 20d sum of overnight gaps (open/prev_close-1), skip-5 (overnight momentum)
  gap_vol_20       : 20d std of overnight gaps (overnight volatility regime)
  trend_r2_60      : R^2 of log-price OLS trend over 60d (trend strength/quality)
  win_rate_60      : fraction of positive days over 60d (momentum consistency)
  vol_ratio_5_60   : 5d realized vol / 60d realized vol (short vol regime)
  vol_ratio_20_60  : 20d realized vol / 60d realized vol
  amihud_20        : mean |ret|/volume over 20d (illiquidity)
  upper_shadow_20  : mean (high-max(open,close))/(high-low) over 20d (selling pressure)
  skew_60          : 60d return skewness
  rsi_14           : classic RSI(14)
  corr_wti_60      : 60d return correlation with WTI (commodity beta)
  corr_ndx_60      : 60d return correlation with NDX (tech beta)
  btc_beta_60      : 60d beta to BTC returns (crypto beta)
  volume_z_20      : 20d mean volume / 60d mean volume (volume expansion)
  macd_hist_20     : MACD(12,26,9) histogram (momentum oscillator)
  stoch_k_14       : %K stochastic oscillator (14d)
  max_dd_60        : 1 - close/rolling_max(close,60) (drawdown depth)
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, GRID, HORIZON, MIN_ASSETS, load_asset, to_grid, safe_div,
    cross_sectional_rank, spearman_ic_matrix, summarize, decay_curve,
    fwd_by_horizon_dict, turnover_10d_rank, coverage_stats,
)

GATE_IC = 0.0070
GATE_ICIR = 0.0840

print("loading assets...", flush=True)
series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series)}", flush=True)
print(f"grid: {len(GRID)} dates {GRID[0]}..{GRID[-1]}", flush=True)

# per-asset derived columns
for s, df in series.items():
    df["ret"] = df["close"].pct_change()
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["hl"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_shadow"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["hl"]
    df["logc"] = np.log(df["close"])

spx_ret = series["SPX"]["ret"]
wti_ret = series["WTI"]["ret"] if "WTI" in series else None
ndx_ret = series["NDX"]["ret"] if "NDX" in series else None
btc_ret = series["BTC"]["ret"] if "BTC" in series else None

def roll_beta(a, b, w, minp=30):
    j = pd.concat([a, b], axis=1, join="outer")
    j.columns = ["a", "b"]
    cov = j["a"].rolling(w, min_periods=minp).cov(j["b"])
    var = j["b"].rolling(w, min_periods=minp).var()
    return (cov / var).reindex(a.index)

def roll_corr(a, b, w, minp=15):
    j = pd.concat([a, b], axis=1, join="outer")
    j.columns = ["a", "b"]
    return j["a"].rolling(w, min_periods=minp).corr(j["b"]).reindex(a.index)

def rmean(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()

def rstd(s, w, minp):
    return s.rolling(w, min_periods=minp).std()

def trend_r2(logc, w=60, minp=40):
    x = np.arange(w, dtype=float)
    def _r2(y):
        if np.isnan(y).sum() > 0:
            return np.nan
        c = np.corrcoef(y, x)[0, 1]
        return c * c
    out = logc.rolling(w, min_periods=minp).apply(_r2, raw=True)
    return out

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta.clip(upper=0.0))
    au = up.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def macd_hist(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    return (line - signal) / close  # normalize by price level

def stoch_k(df, n=14):
    ll = df["low"].rolling(n, min_periods=n).min()
    hh = df["high"].rolling(n, min_periods=n).max()
    return (df["close"] - ll) / (hh - ll).replace(0, np.nan)

def max_dd(close, w=60):
    rollmax = close.rolling(w, min_periods=40).max()
    return 1.0 - close / rollmax

cands = {}

# 1. gap_mom_20 (skip-5 overnight momentum)
c = {}
for s, df in series.items():
    c[s] = df["gap"].rolling(20, min_periods=10).sum().shift(5)
cands["gap_mom_20"] = c

# 2. gap_vol_20
c = {}
for s, df in series.items():
    c[s] = rstd(df["gap"], 20, 10)
cands["gap_vol_20"] = c

# 3. trend_r2_60
c = {s: trend_r2(df["logc"], 60, 40) for s, df in series.items()}
cands["trend_r2_60"] = c

# 4. win_rate_60
c = {}
for s, df in series.items():
    c[s] = (df["ret"] > 0).astype(float).rolling(60, min_periods=30).mean()
cands["win_rate_60"] = c

# 5. vol_ratio_5_60
c = {}
for s, df in series.items():
    v5 = rstd(df["ret"], 5, 3)
    v60 = rstd(df["ret"], 60, 15)
    c[s] = safe_div(v5, v60)
cands["vol_ratio_5_60"] = c

# 6. vol_ratio_20_60
c = {}
for s, df in series.items():
    v20 = rstd(df["ret"], 20, 10)
    v60 = rstd(df["ret"], 60, 15)
    c[s] = safe_div(v20, v60)
cands["vol_ratio_20_60"] = c

# 7. amihud_20
c = {}
for s, df in series.items():
    amih = df["ret"].abs() / df["volume"].replace(0, np.nan)
    c[s] = amih.rolling(20, min_periods=10).mean()
cands["amihud_20"] = c

# 8. upper_shadow_20
c = {s: rmean(df["upper_shadow"], 20, 10) for s, df in series.items()}
cands["upper_shadow_20"] = c

# 9. skew_60
c = {s: df["ret"].rolling(60, min_periods=30).skew() for s, df in series.items()}
cands["skew_60"] = c

# 10. rsi_14
c = {s: rsi(df["close"], 14) for s, df in series.items()}
cands["rsi_14"] = c

# 11. corr_wti_60
c = {}
for s, df in series.items():
    if s == "WTI":
        c[s] = pd.Series(1.0, index=df.index)
    else:
        c[s] = roll_corr(df["ret"], wti_ret, 60, 15)
cands["corr_wti_60"] = c

# 12. corr_ndx_60
c = {}
for s, df in series.items():
    if s == "NDX":
        c[s] = pd.Series(1.0, index=df.index)
    else:
        c[s] = roll_corr(df["ret"], ndx_ret, 60, 15)
cands["corr_ndx_60"] = c

# 13. btc_beta_60
c = {}
for s, df in series.items():
    if s == "BTC":
        c[s] = pd.Series(1.0, index=df.index)
    else:
        c[s] = roll_beta(df["ret"], btc_ret, 60, 30)
cands["btc_beta_60"] = c

# 14. volume_z_20
c = {}
for s, df in series.items():
    v20 = df["volume"].rolling(20, min_periods=10).mean()
    v60 = df["volume"].rolling(60, min_periods=15).mean()
    c[s] = safe_div(v20, v60)
cands["volume_z_20"] = c

# 15. macd_hist_20
c = {s: macd_hist(df["close"]) for s, df in series.items()}
cands["macd_hist_20"] = c

# 16. stoch_k_14
c = {s: stoch_k(df, 14) for s, df in series.items()}
cands["stoch_k_14"] = c

# 17. max_dd_60
c = {s: max_dd(df["close"], 60) for s, df in series.items()}
cands["max_dd_60"] = c

# forward return rank matrices
fwd_ranks = fwd_by_horizon_dict(series)
print("fwd rank matrices ready", flush=True)

results = {}
for name, c in cands.items():
    mat = to_grid(c)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd_ranks[HORIZON])
    if not ics:
        print(f"{name}: NO VALID IC DATES", flush=True)
        results[name] = None
        continue
    dates = np.array(GRID)
    summ = summarize(ics, dates, name, HORIZON)
    ic, icir = summ["ic"], summ["icir"]
    hit = summ["hit"]
    n = summ["n_ic_dates"]
    cov_ad, cov_d8 = coverage_stats(mat)
    tov = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd_ranks)
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    print("=" * 96)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={n} q={q:.5f} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={tov:.3f} GATE={ok}", flush=True)
    print("   regime:", {k: v for k, v in summ["regime"].items()}, flush=True)
    print("   decay:", dec, flush=True)
    results[name] = {"ic": ic, "icir": icir, "q": q, "ok": ok, "hit": hit,
                     "n_ic_dates": n, "regime": summ["regime"], "decay": dec,
                     "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_d8,
                     "turnover_10d_rank": tov}

with open("scripts/miner_2_20270225_batch_explore_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. Passing candidates:", [k for k, v in results.items() if v and v["ok"]])
