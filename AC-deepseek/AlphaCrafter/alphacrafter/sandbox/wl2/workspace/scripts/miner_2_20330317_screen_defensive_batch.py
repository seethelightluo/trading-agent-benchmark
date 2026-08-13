"""miner_2 2033-03-17: screen novel factor candidates (no persistence yet).
Full sample 2020-01-01..2033-03-16 (visible through). 15-asset tradable cross-asset universe.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 on 10d forward returns (daily cross-sectional Spearman).
Macro series truncated at VISIBLE to prevent future leakage.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import (
    ASSETS, GRID, N_GRID, VISIBLE, load_asset, load_macro, safe_div,
    cross_sectional_rank, spearman_ic_matrix, summarize, decay_curve,
    fwd_by_horizon_dict, turnover_10d_rank, library_pairwise_corr,
    coverage_stats, HORIZON, MIN_ASSETS,
)

VIS = pd.Timestamp(VISIBLE)

def load_macro_cut(name):
    s = load_macro(name)
    if s is None:
        return None
    return s[s.index <= VIS.strftime("%Y-%m-%d")]

series = {}
for s in ASSETS:
    df = load_asset(s, days=4000)
    if df is None or len(df) < 200:
        print("SKIP", s); continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    vol = df["volume"].astype(float)
    series[s] = pd.DataFrame({"close": close, "ret": ret, "fwd10": close.shift(-HORIZON) / close - 1.0, "volume": vol})

dxy = load_macro_cut("DXY")
vix = load_macro_cut("VIX")
usdjpy = load_macro_cut("USDJPY")
usdcny = load_macro_cut("USDCNY")
eurusd = load_macro_cut("EURUSD")
print("macro lens:", {k: (None if v is None else len(v)) for k, v in
      [("DXY", dxy), ("VIX", vix), ("USDJPY", usdjpy), ("USDCNY", usdcny), ("EURUSD", eurusd)]})

def roll_beta(a, b, w, minp=30):
    ab = (a * b).rolling(w, min_periods=minp).mean()
    bb = (b * b).rolling(w, min_periods=minp).mean()
    return safe_div(ab, bb)

# equal-weight cross-asset basket return (own-calendar mean)
basket = pd.concat([series[s]["ret"] for s in ASSETS], axis=1).mean(axis=1, skipna=True)

factors = {}

def add_factor(fid, panel):
    factors.setdefault(fid, {}).update(panel)

# 1 vix_corr_60: corr of asset ret with dVIX over 60d (defensive = positive corr)
if vix is not None:
    dvix = vix.pct_change()
    for s, df in series.items():
        add_factor("vix_corr_60", {s: df["ret"].rolling(60, min_periods=30).corr(dvix.reindex(df.index))})

# 2 vix_beta_60: beta of asset ret on dVIX over 60d
if vix is not None:
    dvix = vix.pct_change()
    for s, df in series.items():
        add_factor("vix_beta_60", {s: roll_beta(df["ret"], dvix.reindex(df.index), 60, 30)})

# 3 dd_depth_60: (close - rolling_max60)/rolling_max60
for s, df in series.items():
    mx = df["close"].rolling(60, min_periods=30).max()
    add_factor("dd_depth_60", {s: safe_div(df["close"] - mx, mx)})

# 4 dd_depth_120
for s, df in series.items():
    mx = df["close"].rolling(120, min_periods=60).max()
    add_factor("dd_depth_120", {s: safe_div(df["close"] - mx, mx)})

# 5 drawup_60: (close - min60)/(max60 - min60) recovery position
for s, df in series.items():
    hi = df["close"].rolling(60, min_periods=30).max()
    lo = df["close"].rolling(60, min_periods=30).min()
    add_factor("drawup_60", {s: safe_div(df["close"] - lo, hi - lo)})

# 6 rev5_skip1: -(5d return skipping 1 day) short-term reversal
for s, df in series.items():
    r5 = df["close"] / df["close"].shift(6) - 1.0
    add_factor("rev5_skip1", {s: -r5})

# 7 vol_ratio_5x60: 5d vol / 60d vol
for s, df in series.items():
    v5 = df["ret"].rolling(5, min_periods=3).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    add_factor("vol_ratio_5x60", {s: safe_div(v5, v60)})

# 8 xau_beta_60: beta vs XAU
xau_ret = series["XAU"]["ret"]
for s, df in series.items():
    add_factor("xau_beta_60", {s: roll_beta(df["ret"], xau_ret.reindex(df.index), 60, 30)})

# 9 us10y_beta_60: beta vs US10Y returns
us10y_ret = series["US10Y"]["ret"]
for s, df in series.items():
    add_factor("us10y_beta_60", {s: roll_beta(df["ret"], us10y_ret.reindex(df.index), 60, 30)})

# 10 wti_beta_60: beta vs WTI
wti_ret = series["WTI"]["ret"]
for s, df in series.items():
    add_factor("wti_beta_60", {s: roll_beta(df["ret"], wti_ret.reindex(df.index), 60, 30)})

# 11 mom60_voladj_20: 60d mom (skip5) / 20d vol
for s, df in series.items():
    m = df["close"] / df["close"].shift(65) - 1.0
    v = df["ret"].rolling(20, min_periods=10).std()
    add_factor("mom60_voladj_20", {s: safe_div(m, v)})

# 12 skew_60: realized skewness 60d
for s, df in series.items():
    r = df["ret"]
    mu = r.rolling(60, min_periods=30).mean()
    sd = r.rolling(60, min_periods=30).std()
    m3 = ((r - mu) ** 3).rolling(60, min_periods=30).mean()
    add_factor("skew_60", {s: safe_div(m3, sd ** 3 + 1e-18)})

# 13 autocorr_20: lag-1 autocorr of daily returns 20d
for s, df in series.items():
    r = df["ret"]
    c0 = r.rolling(20, min_periods=10).cov(r)
    c1 = r.rolling(20, min_periods=10).cov(r.shift(1))
    add_factor("autocorr_20", {s: safe_div(c1, c0 + 1e-18)})

# 14 xbeta_ew_60: beta to equal-weight basket
for s, df in series.items():
    add_factor("xbeta_ew_60", {s: roll_beta(df["ret"], basket.reindex(df.index), 60, 30)})

# 15 idio_mom_120: 120d skip5 mom minus basket mom
bmom = pd.concat([series[s]["close"] / series[s]["close"].shift(125) - 1.0 for s in ASSETS], axis=1).mean(axis=1, skipna=True)
for s, df in series.items():
    m = df["close"] / df["close"].shift(125) - 1.0
    add_factor("idio_mom_120", {s: m - bmom.reindex(df.index)})

# 16 corr_breadth_60: mean pairwise corr with all other assets over 60d
rets = {s: series[s]["ret"] for s in ASSETS}
for s, df in series.items():
    parts = []
    for o in ASSETS:
        if o == s:
            continue
        c = df["ret"].rolling(60, min_periods=30).corr(rets[o].reindex(df.index))
        parts.append(c)
    add_factor("corr_breadth_60", {s: pd.concat(parts, axis=1).mean(axis=1, skipna=True)})

# 17 usdcny_beta_60: beta vs USDCNY returns
if usdcny is not None:
    for s, df in series.items():
        add_factor("usdcny_beta_60", {s: roll_beta(df["ret"], usdcny.reindex(df.index).pct_change(), 60, 30)})

# 18 range_compression_20: (max20-min20)/close
for s, df in series.items():
    hi = df["close"].rolling(20, min_periods=10).max()
    lo = df["close"].rolling(20, min_periods=10).min()
    add_factor("range_compression_20", {s: safe_div(hi - lo, df["close"])})

# 19 usdjpy_beta_60: beta vs USDJPY (carry proxy)
if usdjpy is not None:
    for s, df in series.items():
        add_factor("usdjpy_beta_60", {s: roll_beta(df["ret"], usdjpy.reindex(df.index).pct_change(), 60, 30)})

# 20 updown_asym_60: (up_mean - dn_mean)/(up_mean+dn_mean)
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0.0).rolling(60, min_periods=30).mean()
    dn = (-r.clip(upper=0.0)).rolling(60, min_periods=30).mean()
    add_factor("updown_asym_60", {s: safe_div(up - dn, up + dn + 1e-18)})

def to_grid_panel(panel):
    out = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s in panel:
            out[:, j] = panel[s].reindex(GRID).values
    return out

dates = np.array(GRID)
fwd = fwd_by_horizon_dict(series)
fwd10 = fwd["10"]

results = {}
for fid, panel in factors.items():
    mat = to_grid_panel(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(fid, "NO IC DATES"); continue
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
    rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["decay"] = decay_curve(rank_mat, fwd)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    reg250 = s["regime"].get("last250", {})
    results[fid] = s
    print(f"{fid:24s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} turn={s['turnover_10d_rank']:.3f} "
          f"cov={s['coverage']:.3f} maxrho={max_rho:.3f}({rho_name}) l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("idx", "icv")} for k, v in results.items()},
          open("scripts/miner_2_20330317_screen_defensive_batch.json", "w"), indent=1, default=str)
print("DONE. saved scripts/miner_2_20330317_screen_defensive_batch.json")
