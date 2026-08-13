"""miner_1 2031-03-06: baseline revalidation of the full factor library on data through 2031-03-05.
Recomputes every library factor from raw prices, reports IC/ICIR/hit/coverage/decay/maxcorr.
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (same 15-instrument universe).
Also computes latest-250d (last250) and last-750d windows for drift tracking.
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

DAYS = 3400


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


series = asset_series_full()
print("assets with data:", sorted(series.keys()))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)
print("grid dates:", len(dates), "from", dates[0], "to", dates[-1])

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
for s, df in series.items():
    r = df["ret"]
    th = 0.5 * r.rolling(20, min_periods=10).std()
    factors.setdefault("calmness_20", {})[s] = (r.abs() < th).rolling(20, min_periods=10).mean()
for s, df in series.items():
    rng = (df["high"] - df["low"])
    cp = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    factors.setdefault("close_pos_20", {})[s] = cp.rolling(20, min_periods=10).mean()
for s, df in series.items():
    roll_max = df["close"].rolling(60, min_periods=20).max()
    dd = df["close"] / roll_max - 1.0
    days = (dd < -0.01).astype(float)
    factors.setdefault("days_since_high_60", {})[s] = -days.rolling(60, min_periods=20).sum()
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("downbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta_cond(df["ret"], spx.reindex(df.index).pct_change(), 60, 30, cond=spx.reindex(df.index).pct_change() < 0)
    factors.setdefault("downbeta_spx_60", {})[s] = -b
for s, df in series.items():
    b = roll_beta_cond(df["ret"], dxy.reindex(df.index).pct_change(), 60, 30)
    m = dxy.reindex(df.index) / dxy.reindex(df.index).shift(20) - 1.0
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = b * m
for s, df in series.items():
    r = df["ret"]
    g = r.clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-r).clip(lower=0).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = g / l.replace(0, np.nan)
for s, df in series.items():
    idr = df["close"] / df["open"] - 1.0
    factors.setdefault("intraday_drift_20", {})[s] = idr.rolling(20, min_periods=10).mean()
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("lagbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta_cond(df["ret"], spx.reindex(df.index).pct_change().shift(1), 60, 30)
    factors.setdefault("lagbeta_spx_60", {})[s] = -b
for s, df in series.items():
    r = df["ret"].fillna(0.0)
    cg = 0.0
    vals = []
    for v in r.values:
        cg = cg + 1 if v > 0 else 0
        vals.append(cg)
    cs = pd.Series(vals, index=df.index)
    factors.setdefault("max_consec_gain_20", {})[s] = cs.rolling(20, min_periods=10).max()
for s, df in series.items():
    r = df["ret"].fillna(0.0)
    cl = 0.0
    vals = []
    for v in r.values:
        cl = cl + 1 if v < 0 else 0
        vals.append(cl)
    cs = pd.Series(vals, index=df.index)
    factors.setdefault("max_consec_loss_20", {})[s] = -cs.rolling(20, min_periods=10).max()
for s, df in series.items():
    factors.setdefault("mom_10d_skip5", {})[s] = df["close"] / df["close"].shift(5) - 1.0
for s, df in series.items():
    factors.setdefault("mom_20d_skip5", {})[s] = df["close"] / df["close"].shift(15) - 1.0
for s, df in series.items():
    factors.setdefault("mom_120d_skip5", {})[s] = df["close"] / df["close"].shift(115) - 1.0
for s, df in series.items():
    factors.setdefault("mom_180d_skip5", {})[s] = df["close"] / df["close"].shift(175) - 1.0
for s, df in series.items():
    m = df["close"] / df["close"].shift(20) - 1.0
    v = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("mom20_volproxy60", {})[s] = m / v.replace(0, np.nan)
for s, df in series.items():
    hi = df["high"].rolling(252, min_periods=120).max()
    lo = df["low"].rolling(252, min_periods=120).min()
    factors.setdefault("range_pos_252", {})[s] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("spx_corr60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    c = df["ret"].rolling(60, min_periods=30).corr(spx.reindex(df.index).pct_change())
    factors.setdefault("spx_corr60", {})[s] = c
for s, df in series.items():
    b = roll_beta_cond(df["ret"], usdjpy.reindex(df.index).pct_change(), 120, 60)
    m = usdjpy.reindex(df.index) / usdjpy.reindex(df.index).shift(60) - 1.0
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = b * m
for s, df in series.items():
    b = roll_beta_cond(df["ret"], -vix.reindex(df.index).pct_change(), 60, 30)
    m = vix.reindex(df.index) / vix.reindex(df.index).shift(20) - 1.0
    factors.setdefault("vix_beta_cond_60x20", {})[s] = b * m
for s, df in series.items():
    v = df["ret"].rolling(20, min_periods=10).std()
    factors.setdefault("vol_of_vol20x60", {})[s] = v.rolling(60, min_periods=30).std()
for s, df in series.items():
    v = df["ret"].rolling(20, min_periods=10).std()
    vm = v.rolling(60, min_periods=30).mean()
    factors.setdefault("volcluster_60", {})[s] = v / vm.replace(0, np.nan)
for s, df in series.items():
    m = df["close"] / df["close"].shift(30) - 1.0
    v = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("mom30_vol60", {})[s] = m / v.replace(0, np.nan)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if not ics:
        print(fid, "NO IC DATES")
        continue
    s = summarize(ics, dates, fid, HORIZON)
    rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["decay"] = decay_curve(rank_mat, fwd_by_h)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    # last250 window drift
    icv = np.array([v for _, v in ics])
    idx = np.array([t for t, _ in ics])
    if len(icv) >= 250:
        m = idx >= len(dates) - 250
        s["last250_ic"] = round(float(np.mean(icv[m])), 4)
        s["last250_icir"] = round(float(np.mean(icv[m]) / np.std(icv[m])), 3) if np.std(icv[m]) > 0 else 0.0
    if len(icv) >= 750:
        m = idx >= len(dates) - 750
        s["last750_ic"] = round(float(np.mean(icv[m])), 4)
        s["last750_icir"] = round(float(np.mean(icv[m]) / np.std(icv[m])), 3) if np.std(icv[m]) > 0 else 0.0
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    print(f"{fid:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_ic_dates']} "
          f"cov={s['coverage']:.3f} turn={s['turnover_10d_rank']:.3f} maxrho={max_rho:.3f} "
          f"l250={s.get('last250_ic','NA')}/{s.get('last250_icir','NA')} decay={s['decay']} -> {'PASS' if s['ok'] else 'fail'}")

json.dump(results, open("scripts/miner_1_20310306_revalidate_baseline.json", "w"), indent=1, default=str)
print("\nSaved scripts/miner_1_20310306_revalidate_baseline.json")
