"""miner_3 2028-10-05: full library revalidation with data through 2028-10-04.
Recomputes every library factor from raw prices and reports IC/ICIR/hit/coverage/turnover/decay/maxcorr.
Gates: |IC|>=0.0070, |ICIR|>=0.0840."""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, asset_series, to_grid, load_macro,
    roll_mean, roll_std, safe_div, cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats, HORIZON, MIN_ASSETS,
)

OUT = "scripts/miner_3_20281005_revalidate_results.json"
series = asset_series()
print("assets with data:", sorted(series.keys()))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
usdjpy = load_macro("USDJPY")
vix = load_macro("VIX")


def rstd(x, w, minp):
    return x.rolling(w, min_periods=minp).std()


def roll_beta_cond(asset_ret, ref_ret, w, minp, cond=None):
    """rolling beta of asset_ret vs ref_ret; cond: boolean mask on ref_ret."""
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


def rolling_corr(a, b, w, minp):
    return a.rolling(w, min_periods=minp).corr(b)


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
    c = df["close"]
    roll_max = c.rolling(60, min_periods=40).max()
    since = np.nan
    vals = []
    for i in range(len(c)):
        if i < 39 or not np.isfinite(roll_max.iloc[i]):
            vals.append(np.nan); continue
        mx = roll_max.iloc[i]
        j = i
        while j >= 0 and j > i - 60:
            if c.iloc[j] == mx:
                break
            j -= 1
        vals.append(i - j)
    factors.setdefault("days_since_high_60", {})[s] = pd.Series(vals, index=c.index)

# 4 downbeta_spx_60
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("downbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    cond = spx.reindex(df.index).pct_change() < 0
    factors.setdefault("downbeta_spx_60", {})[s] = roll_beta_cond(
        df["ret"], spx.reindex(df.index).pct_change(), 60, 15, cond)

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
    l = r.clip(upper=0).rolling(20, min_periods=10).mean().abs() + 1e-9
    factors.setdefault("gain_loss_20", {})[s] = g / l

# 7 intraday_drift_20
for s, df in series.items():
    factors.setdefault("intraday_drift_20", {})[s] = (df["close"] / df["open"] - 1.0).rolling(20, min_periods=10).mean()

# 8 lagbeta_spx_60
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("lagbeta_spx_60", {})[s] = pd.Series(0.0, index=df.index)
        continue
    factors.setdefault("lagbeta_spx_60", {})[s] = roll_beta_cond(
        df["ret"], spx.reindex(df.index).pct_change().shift(1), 60, 30)

# 9 max_consec_gain_20
for s, df in series.items():
    r = (df["ret"] > 0).astype(float).values
    vals = np.full(len(r), np.nan)
    for i in range(len(r)):
        if i < 10 or not np.isfinite(r[i]):
            continue
        mx = 0.0; cur = 0.0
        for j in range(max(0, i - 20), i + 1):
            if r[j] == 1.0:
                cur += 1; mx = max(mx, cur)
            else:
                cur = 0.0
        vals[i] = mx
    factors.setdefault("max_consec_gain_20", {})[s] = pd.Series(vals, index=df.index)

# 10 max_consec_loss_20
for s, df in series.items():
    r = (df["ret"] < 0).astype(float).values
    vals = np.full(len(r), np.nan)
    for i in range(len(r)):
        if i < 10 or not np.isfinite(r[i]):
            continue
        mx = 0.0; cur = 0.0
        for j in range(max(0, i - 20), i + 1):
            if r[j] == 1.0:
                cur += 1; mx = max(mx, cur)
            else:
                cur = 0.0
        vals[i] = mx
    factors.setdefault("max_consec_loss_20", {})[s] = pd.Series(vals, index=df.index)

# 11-15 momentum family
for lb, sk in [(20, 5), (10, 5), (120, 5), (180, 5)]:
    fid = f"mom_{lb}d_skip5"
    for s, df in series.items():
        c = df["close"]
        factors.setdefault(fid, {})[s] = c.shift(sk) / c.shift(sk + lb) - 1.0

# mom20_volproxy60
for s, df in series.items():
    c = df["close"]
    mom = c.shift(5) / c.shift(25) - 1.0
    vp = c.pct_change().rolling(60, min_periods=15).std()
    factors.setdefault("mom20_volproxy60", {})[s] = mom / (1.0 + vp.abs())

# mom30_vol60
for s, df in series.items():
    c = df["close"]
    mom = c.shift(5) / c.shift(35) - 1.0
    v = c.pct_change().rolling(60, min_periods=15).std()
    factors.setdefault("mom30_vol60", {})[s] = mom / v

# 16 range_pos_252
for s, df in series.items():
    c = df["close"]
    lo = c.rolling(252, min_periods=30).min()
    hi = c.rolling(252, min_periods=30).max()
    factors.setdefault("range_pos_252", {})[s] = (c - lo) / (hi - lo)

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

# 22 gain_loss_20 alt not needed

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
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"last250={s['regime'].get('last250',{}).get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT)
