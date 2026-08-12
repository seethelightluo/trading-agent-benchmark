"""miner_3 2029-06-14: full library revalidation with data through 2029-06-13.
Recomputes every library factor from raw prices and reports IC/ICIR/hit/coverage/turnover/decay/maxcorr.
Gates: |IC|>=0.0070, |ICIR|>=0.0840. Full 3400-day window to cover master grid from 2020-01-01.
"""
import sys, json, glob, os
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

OUT = "scripts/miner_3_20290614_revalidate_results.json"


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=3400)
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


series = asset_series_full()
print("assets with data:", sorted(series.keys()))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
usdjpy = load_macro("USDJPY")
vix = load_macro("VIX")
eurusd = load_macro("EURUSD")
usdcny = load_macro("USDCNY")


def roll_beta_cond(asset_ret, ref_ret, w, minp, cond=None):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    c = None if cond is None else cond.reindex(asset_ret.index).values.astype(bool)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        if c is not None:
            m = c[seg]
            x = x[m]; y = y[m]
        if len(x) < minp or np.std(x) < 1e-12:
            continue
        beta = np.cov(x, y)[0, 1] / np.var(x)
        if np.isfinite(beta):
            out.iloc[i] = beta
    return out


factors = {}

# 1 calmness_20
for s, df in series.items():
    r = df["ret"]
    th = 0.5 * r.rolling(20, min_periods=10).std()
    factors.setdefault("calmness_20", {})[s] = (r.abs() < th).rolling(20, min_periods=10).mean()

# 2 close_pos_20
for s, df in series.items():
    rng = (df["high"] - df["low"])
    cp = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    factors.setdefault("close_pos_20", {})[s] = cp.rolling(20, min_periods=10).mean()

# 3 days_since_high_60
for s, df in series.items():
    roll_max = df["close"].rolling(60, min_periods=20).max()
    dd = df["close"] / roll_max - 1.0
    days = (dd < -0.01).astype(float)
    days = days.rolling(60, min_periods=20).sum()
    factors.setdefault("days_since_high_60", {})[s] = -days

# 4 downbeta_spx_60
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("downbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta_cond(df["ret"], spx.reindex(df.index).pct_change(), 60, 30, cond=spx.reindex(df.index).pct_change() < 0)
    factors.setdefault("downbeta_spx_60", {})[s] = -b

# 5 dxy_beta_cond_60x20
for s, df in series.items():
    if dxy is None:
        continue
    beta = roll_beta_cond(df["ret"], dxy.reindex(df.index).pct_change(), 60, 30)
    m = dxy.reindex(df.index) / dxy.reindex(df.index).shift(20) - 1.0
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = beta * m

# 6 gain_loss_20
for s, df in series.items():
    r = df["ret"]
    g = r.clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-r.clip(upper=0)).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = g - l

# 7 intraday_drift_20
for s, df in series.items():
    dr = df["close"] / df["open"] - 1.0
    factors.setdefault("intraday_drift_20", {})[s] = dr.rolling(20, min_periods=10).mean()

# 8 lagbeta_spx_60
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("lagbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta_cond(df["ret"], spx.reindex(df.index).pct_change().shift(1), 60, 30)
    factors.setdefault("lagbeta_spx_60", {})[s] = -b

# 9 max_consec_gain_20
for s, df in series.items():
    r = df["ret"]
    pg = (r > 0).astype(int)
    grp = (pg.diff() != 0).cumsum()
    streak = pg.groupby(grp).cumsum()
    factors.setdefault("max_consec_gain_20", {})[s] = streak.rolling(20, min_periods=10).max()

# 10 max_consec_loss_20
for s, df in series.items():
    r = df["ret"]
    pl = (r < 0).astype(int)
    grp = (pl.diff() != 0).cumsum()
    streak = pl.groupby(grp).cumsum()
    factors.setdefault("max_consec_loss_20", {})[s] = -streak.rolling(20, min_periods=10).max()

# 11 mom_10d_skip5
for s, df in series.items():
    factors.setdefault("mom_10d_skip5", {})[s] = df["close"] / df["close"].shift(15) - 1.0

# 12 mom_20d_skip5
for s, df in series.items():
    factors.setdefault("mom_20d_skip5", {})[s] = df["close"] / df["close"].shift(25) - 1.0

# 13 mom_120d_skip5
for s, df in series.items():
    factors.setdefault("mom_120d_skip5", {})[s] = df["close"] / df["close"].shift(125) - 1.0

# 14 mom_180d_skip5
for s, df in series.items():
    factors.setdefault("mom_180d_skip5", {})[s] = df["close"] / df["close"].shift(185) - 1.0

# 15 mom20_volproxy60
for s, df in series.items():
    m = df["close"] / df["close"].shift(20) - 1.0
    v = df["close"].pct_change().rolling(60, min_periods=30).std()
    factors.setdefault("mom20_volproxy60", {})[s] = pd.Series(safe_div(m, v), index=df.index)

# 16 range_pos_252
for s, df in series.items():
    c = df["close"]
    lo = c.rolling(252, min_periods=60).min()
    hi = c.rolling(252, min_periods=60).max()
    factors.setdefault("range_pos_252", {})[s] = pd.Series(safe_div(c - lo, hi - lo), index=df.index)

# 17 spx_corr60
for s, df in series.items():
    a = df["ret"]
    b = spx.reindex(df.index).pct_change()
    factors.setdefault("spx_corr60", {})[s] = a.rolling(60, min_periods=15).corr(b)

# 18 usdjpy_beta_cond_120x60
for s, df in series.items():
    if usdjpy is None:
        continue
    beta = roll_beta_cond(df["ret"], usdjpy.reindex(df.index).pct_change(), 120, 60)
    m = usdjpy.reindex(df.index) / usdjpy.reindex(df.index).shift(60) - 1.0
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = beta * m

# 19 vix_beta_cond_60x20
for s, df in series.items():
    if vix is None:
        continue
    beta = roll_beta_cond(df["ret"], vix.reindex(df.index).pct_change(), 60, 30)
    m = vix.reindex(df.index) / vix.reindex(df.index).shift(20) - 1.0
    factors.setdefault("vix_beta_cond_60x20", {})[s] = -beta * m

# 20 vol_of_vol20x60
for s, df in series.items():
    rv = df["ret"].rolling(20, min_periods=5).std()
    factors.setdefault("vol_of_vol20x60", {})[s] = rv.rolling(60, min_periods=15).std()

# 21 volcluster_60
for s, df in series.items():
    ar = df["ret"].abs()
    factors.setdefault("volcluster_60", {})[s] = ar.rolling(60, min_periods=40).corr(ar.shift(1))

# 22 mom30_vol60
for s, df in series.items():
    m = df["close"] / df["close"].shift(30) - 1.0
    v = df["close"].pct_change().rolling(60, min_periods=30).std()
    factors.setdefault("mom30_vol60", {})[s] = pd.Series(safe_div(m, v), index=df.index)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
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
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT)
