"""miner_3 2032-08-19: screen NOVEL factor candidates batch A (not in library).
Data visible through 2032-08-18 (previous completed trading day).
Conventions match miner_3_20260813_lib: per-asset own calendar, h=10 Spearman rank IC,
>=8 valid assets per date. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
Also reports max_abs_library_correlation vs factors/*.signal.npy artifacts.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, to_grid, load_macro,
    safe_div, cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats, HORIZON, MIN_ASSETS,
)

OUT = "scripts/miner_3_20320819_screen_batchA.json"
DAYS = 4000


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=DAYS)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        ret = close.pct_change()
        fwd = close.shift(-HORIZON) / close - 1.0
        d = pd.DataFrame({
            "close": close, "ret": ret, "fwd10": fwd,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float), "volume": df["volume"].astype(float),
        })
        out[s] = d
    return out


def roll_beta(asset_ret, ref_ret, w, minp):
    """rolling beta of asset_ret on ref_ret (own calendar), plain (unconditional)."""
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < minp:
            continue
        xv = x[m]; yv = y[m]
        sd = xv.std()
        if sd < 1e-12:
            continue
        out.iloc[i] = np.cov(xv, yv)[0, 1] / xv.var()
    return out


series = asset_series_full()
print("assets with data:", sorted(series.keys()), flush=True)
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
vix = load_macro("VIX")
us10y = series.get("US10Y")
xau = series.get("XAU")
cn10y = series.get("CN10Y")

factors = {}

# 1 sharpe_60: 60d momentum scaled by 60d realized vol (risk-adjusted medium momentum)
for s, df in series.items():
    m = df["close"] / df["close"].shift(60) - 1.0
    v = df["ret"].rolling(60, min_periods=30).std()
    factors["sharpe_60"][s] = pd.Series(safe_div(m, v), index=df.index)

# 2 ma_slope_20x60: (MA20/MA60 - 1) trend slope
for s, df in series.items():
    ma20 = df["close"].rolling(20, min_periods=10).mean()
    ma60 = df["close"].rolling(60, min_periods=30).mean()
    factors["ma_slope_20x60"][s] = pd.Series(safe_div(ma20, ma60) - 1.0, index=df.index)

# 3 downside_ratio_60: 60d downside deviation / 60d total std (resilience)
for s, df in series.items():
    r = df["ret"]
    dd = (-r.clip(upper=0)).rolling(60, min_periods=30).mean()
    dd = np.sqrt(np.maximum(dd, 0.0))
    tot = r.rolling(60, min_periods=30).std()
    factors["downside_ratio_60"][s] = pd.Series(safe_div(dd, tot), index=df.index)

# 4 ret5d_rev: -5d return (short-term reversal)
for s, df in series.items():
    factors["ret5d_rev"][s] = -(df["close"] / df["close"].shift(5) - 1.0)

# 5 time_above_ma_60: fraction of last 60 days close > MA20 (trend duration)
for s, df in series.items():
    ma20 = df["close"].rolling(20, min_periods=10).mean()
    above = (df["close"] > ma20).astype(float)
    factors["time_above_ma_60"][s] = above.rolling(60, min_periods=30).mean()

# 6 vix_beta_60: plain 60d beta to VIX pct changes (higher = more VIX-sensitive = risky)
for s, df in series.items():
    if vix is None:
        continue
    factors["vix_beta_60"][s] = roll_beta(df["ret"], vix.reindex(df.index).pct_change(), 60, 30)

# 7 xau_beta_60: 60d beta to XAU returns (safe-haven beta)
for s, df in series.items():
    if xau is None:
        continue
    factors["xau_beta_60"][s] = roll_beta(df["ret"], xau["ret"].reindex(df.index), 60, 30)

# 8 us10y_beta_60: 60d beta to US10Y pct changes (bond-proxy beta)
for s, df in series.items():
    if us10y is None:
        continue
    factors["us10y_beta_60"][s] = roll_beta(df["ret"], us10y["ret"].reindex(df.index), 60, 30)

# 9 vol_ratio_20x60: 20d vol / 60d vol (volatility expansion/contraction)
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    factors["vol_ratio_20x60"][s] = pd.Series(safe_div(v20, v60), index=df.index)

# 10 dxy_beta_60: plain 60d beta to DXY pct changes (dollar beta)
for s, df in series.items():
    if dxy is None:
        continue
    factors["dxy_beta_60"][s] = roll_beta(df["ret"], dxy.reindex(df.index).pct_change(), 60, 30)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(fid, "NO IC DATES", flush=True)
        continue
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
    s["decay"] = decay_curve(rank_mat, fwd_by_h)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    reg250 = s["regime"].get("last250", {})
    print(f"{fid:22s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT)
