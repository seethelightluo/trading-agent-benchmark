"""miner_2 2033-02-03: screen novel factor candidates (no persistence).
Full sample 2020-01-01..2033-02-02 (visible through), 15-asset tradable cross-asset universe.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840. Report regime + last250 for robustness.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import (
    ASSETS, GRID, N_GRID, load_asset, load_macro, safe_div,
    cross_sectional_rank, spearman_ic_matrix, summarize, decay_curve,
    fwd_by_horizon_dict, turnover_10d_rank, library_pairwise_corr,
    coverage_stats, HORIZON, MIN_ASSETS,
)

series = {}
for s in ASSETS:
    df = load_asset(s, days=4000)
    if df is None or len(df) < 200:
        print("SKIP", s)
        continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    vol = df["volume"].astype(float)
    series[s] = pd.DataFrame({"close": close, "ret": ret, "fwd10": close.shift(-HORIZON) / close - 1.0, "volume": vol})

dxy = load_macro("DXY")
vix = load_macro("VIX")
usdjpy = load_macro("USDJPY")
usdcny = load_macro("USDCNY")
eurusd = load_macro("EURUSD")

def roll_beta(a, b, w, minp=30):
    """rolling beta of a on b (both pd.Series aligned by index, own calendar)."""
    ab = (a * b).rolling(w, min_periods=minp).mean()
    aa = (b * b).rolling(w, min_periods=minp).mean()
    return safe_div(ab, aa)

# equal-weight cross-asset basket return (mean of available asset returns per own-calendar date)
basket = pd.concat([series[s]["ret"] for s in ASSETS], axis=1).mean(axis=1, skipna=True)

factors = {}

# 1 skew_60: realized skewness of daily returns over 60d
for s, df in series.items():
    r = df["ret"]
    mu = r.rolling(60, min_periods=30).mean()
    sd = r.rolling(60, min_periods=30).std()
    m3 = ((r - mu) ** 3).rolling(60, min_periods=30).mean()
    factors.setdefault("skew_60", {})[s] = pd.Series(safe_div(m3, sd ** 3 + 1e-18), index=df.index)

# 2 autocorr_20: lag-1 autocorrelation of daily returns over 20d
for s, df in series.items():
    r = df["ret"]
    c0 = r.rolling(20, min_periods=10).cov(r)
    c1 = r.rolling(20, min_periods=10).cov(r.shift(1))
    factors.setdefault("autocorr_20", {})[s] = pd.Series(safe_div(c1, c0 + 1e-18), index=df.index)

# 3 xbeta_ew_60: beta to equal-weight cross-asset basket (systematic co-movement)
for s, df in series.items():
    b = roll_beta(df["ret"], basket.reindex(df.index), 60, 30)
    factors.setdefault("xbeta_ew_60", {})[s] = pd.Series(b, index=df.index)

# 4 idio_mom_120: 120d skip5 momentum minus basket 120d momentum (relative strength)
bmom = pd.concat([series[s]["close"] / series[s]["close"].shift(125) - 1.0 for s in ASSETS], axis=1).mean(axis=1, skipna=True)
for s, df in series.items():
    m = df["close"] / df["close"].shift(125) - 1.0
    factors.setdefault("idio_mom_120", {})[s] = pd.Series(m - bmom.reindex(df.index), index=df.index)

# 5 dur_beta_us10y_60: rolling beta of asset returns vs US10Y return (rate sensitivity)
us10y_ret = series["US10Y"]["ret"]
for s, df in series.items():
    b = roll_beta(df["ret"], us10y_ret.reindex(df.index), 60, 30)
    factors.setdefault("dur_beta_us10y_60", {})[s] = pd.Series(b, index=df.index)

# 6 vol_ratio_20x60: 20d realized vol / 60d realized vol (vol term-structure tilt)
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("vol_ratio_20x60", {})[s] = pd.Series(safe_div(v20, v60), index=df.index)

# 7 updown_asym_60: upside capture minus downside capture asymmetry
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0.0).rolling(60, min_periods=30).mean()
    dn = (-r.clip(upper=0.0)).rolling(60, min_periods=30).mean()
    factors.setdefault("updown_asym_60", {})[s] = pd.Series(safe_div(up - dn, up + dn + 1e-18), index=df.index)

# 8 vol_volume_corr_60: correlation between |ret| and volume over 60d
for s, df in series.items():
    ar = df["ret"].abs()
    vv = df["volume"]
    factors.setdefault("vol_volume_corr_60", {})[s] = ar.rolling(60, min_periods=30).corr(vv)

# 9 drawdown_depth_120: (close - rolling_max 120)/rolling_max 120 (current drawdown depth)
for s, df in series.items():
    mx = df["close"].rolling(120, min_periods=60).max()
    factors.setdefault("drawdown_depth_120", {})[s] = pd.Series(safe_div(df["close"] - mx, mx), index=df.index)

# 10 wti_beta_60: rolling beta vs WTI (commodity beta)
wti_ret = series["WTI"]["ret"]
for s, df in series.items():
    b = roll_beta(df["ret"], wti_ret.reindex(df.index), 60, 30)
    factors.setdefault("wti_beta_60", {})[s] = pd.Series(b, index=df.index)

# 11 usdcny_beta_60: rolling beta vs USDCNY (China FX sensitivity)
if usdcny is not None:
    for s, df in series.items():
        b = roll_beta(df["ret"], usdcny.reindex(df.index).pct_change(), 60, 30)
        factors.setdefault("usdcny_beta_60", {})[s] = pd.Series(b, index=df.index)

# 12 range_compression_20: (rolling_max20 - rolling_min20)/close (bandwidth)
for s, df in series.items():
    hi = df["close"].rolling(20, min_periods=10).max()
    lo = df["close"].rolling(20, min_periods=10).min()
    factors.setdefault("range_compression_20", {})[s] = pd.Series(safe_div(hi - lo, df["close"]), index=df.index)

dates = np.array(GRID)
fwd = fwd_by_horizon_dict(series)
fwd10 = fwd["10"]

results = {}
for fid, panel in factors.items():
    mat = to_grid_panel(panel) if False else None

def to_grid_panel(panel):
    out = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s in panel:
            out[:, j] = panel[s].reindex(GRID).values
    return out

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
          open("scripts/miner_2_20330203_screen_results.json", "w"), indent=1, default=str)
print("DONE. saved scripts/miner_2_20330203_screen_results.json")
