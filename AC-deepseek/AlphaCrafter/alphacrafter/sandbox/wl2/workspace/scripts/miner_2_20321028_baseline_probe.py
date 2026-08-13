"""miner_2 2032-10-28 baseline: recompute 22-factor library on current full grid
(2020-01-01..2032-10-27) and probe data availability to guide novel-factor search.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, to_grid, load_macro,
    safe_div, cross_sectional_rank, spearman_ic_matrix,
    summarize, HORIZON, MIN_ASSETS,
)

DAYS = 4200


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=DAYS)
        if df is None or len(df) < 100:
            print("NO DATA:", s)
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
dates = np.array(GRID)
print("grid rows:", len(GRID), "dates:", GRID[0], "..", GRID[-1])

# probe per-asset data ranges + volume sanity + zero-return stretches
for s in sorted(series.keys()):
    df = series[s]
    nz = (df["ret"].abs() < 1e-12).mean()
    vol_nz = (df["volume"] > 0).mean() if df["volume"].notna().any() else float("nan")
    print(f"{s:10s} rows={len(df):5d} first={df.index[0]} last={df.index[-1]} zero_ret_frac={nz:.3f} vol_pos_frac={vol_nz:.3f}")

spx = series["SPX"]["close"]
dxy = load_macro("DXY"); usdjpy = load_macro("USDJPY"); vix = load_macro("VIX")
usdcny = load_macro("USDCNY"); eurusd = load_macro("EURUSD")
print("macro loaded:", {k: (v is not None) for k, v in
      [("DXY", dxy), ("USDJPY", usdjpy), ("VIX", vix), ("USDCNY", usdcny), ("EURUSD", eurusd)]})
if dxy is not None:
    print("DXY range:", dxy.index[0], dxy.index[-1], "rows:", len(dxy))


def roll_beta_cond(asset_ret, ref_ret, w, minp, cond=None):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    c = None if cond is None else cond.reindex(asset_ret.index).values.astype(bool)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        m = ~(np.isnan(x) | np.isnan(y))
        if c is not None:
            m = m & c[seg]
        if m.sum() < minp:
            continue
        xv = x[m]; yv = y[m]
        sd = xv.std()
        if sd < 1e-12:
            continue
        out.iloc[i] = np.cov(xv, yv)[0, 1] / xv.var()
    return out


factors = {}
# library factors (22)
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"]
    factors.setdefault("calmness_20", {})[s] = 1.0 - rng.rolling(20, min_periods=10).mean()
for s, df in series.items():
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = (df["close"] - df["low"]) / rng
    factors.setdefault("close_pos_20", {})[s] = pos.rolling(20, min_periods=10).mean()
for s, df in series.items():
    rl = df["close"].rolling(60, min_periods=30).max()
    dsh = (df["close"] == rl).astype(float)
    out = pd.Series(np.nan, index=df.index)
    cnt = 0.0
    for i in range(len(df)):
        if dsh.iloc[i] == 1.0:
            cnt = 0.0
        else:
            cnt += 1.0
        out.iloc[i] = cnt
    factors.setdefault("days_since_high_60", {})[s] = out
for s, df in series.items():
    spxr = spx.pct_change()
    factors.setdefault("downbeta_spx_60", {})[s] = roll_beta_cond(df["ret"], spxr, 60, 30, cond=spxr < 0)
for s, df in series.items():
    if dxy is None:
        continue
    beta = roll_beta_cond(df["ret"], dxy.reindex(df.index).pct_change(), 60, 30)
    m = dxy.reindex(df.index) / dxy.reindex(df.index).shift(20) - 1.0
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = beta * m
for s, df in series.items():
    g = df["ret"].clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-df["ret"].clip(upper=0)).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = pd.Series(safe_div(g - l, g + l), index=df.index)
for s, df in series.items():
    drift = df["close"] / df["open"] - 1.0
    factors.setdefault("intraday_drift_20", {})[s] = drift.rolling(20, min_periods=10).mean()
for s, df in series.items():
    spxr = spx.pct_change()
    factors.setdefault("lagbeta_spx_60", {})[s] = roll_beta_cond(df["ret"], spxr.shift(1), 60, 30)
for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    out = pd.Series(np.nan, index=df.index)
    cnt = 0.0
    for i in range(len(df)):
        if pos.iloc[i] == 1.0:
            cnt += 1.0
        else:
            cnt = 0.0
        out.iloc[i] = cnt
    factors.setdefault("max_consec_gain_20", {})[s] = out.rolling(20, min_periods=10).max()
for s, df in series.items():
    neg = (df["ret"] < 0).astype(float)
    out = pd.Series(np.nan, index=df.index)
    cnt = 0.0
    for i in range(len(df)):
        if neg.iloc[i] == 1.0:
            cnt += 1.0
        else:
            cnt = 0.0
        out.iloc[i] = cnt
    factors.setdefault("max_consec_loss_20", {})[s] = out.rolling(20, min_periods=10).max()
for s, df in series.items():
    factors.setdefault("mom_10d_skip5", {})[s] = df["close"] / df["close"].shift(15) - 1.0
for s, df in series.items():
    factors.setdefault("mom_20d_skip5", {})[s] = df["close"] / df["close"].shift(25) - 1.0
for s, df in series.items():
    factors.setdefault("mom_120d_skip5", {})[s] = df["close"] / df["close"].shift(125) - 1.0
for s, df in series.items():
    factors.setdefault("mom_180d_skip5", {})[s] = df["close"] / df["close"].shift(185) - 1.0
for s, df in series.items():
    m = df["close"] / df["close"].shift(20) - 1.0
    v = df["close"].pct_change().rolling(60, min_periods=30).std()
    factors.setdefault("mom20_volproxy60", {})[s] = pd.Series(safe_div(m, v), index=df.index)
for s, df in series.items():
    lo = df["close"].rolling(252, min_periods=120).min()
    hi = df["close"].rolling(252, min_periods=120).max()
    factors.setdefault("range_pos_252", {})[s] = pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)
for s, df in series.items():
    spxr = spx.pct_change()
    a = df["ret"]; b = spxr.reindex(df.index)
    factors.setdefault("spx_corr60", {})[s] = a.rolling(60, min_periods=30).corr(b)
for s, df in series.items():
    if usdjpy is None:
        continue
    beta = roll_beta_cond(df["ret"], usdjpy.reindex(df.index).pct_change(), 120, 60)
    m = usdjpy.reindex(df.index) / usdjpy.reindex(df.index).shift(60) - 1.0
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = beta * m
for s, df in series.items():
    if vix is None:
        continue
    beta = roll_beta_cond(df["ret"], vix.reindex(df.index).pct_change(), 60, 30)
    m = vix.reindex(df.index) / vix.reindex(df.index).shift(20) - 1.0
    factors.setdefault("vix_beta_cond_60x20", {})[s] = -beta * m
for s, df in series.items():
    rv = df["ret"].rolling(20, min_periods=5).std()
    factors.setdefault("vol_of_vol20x60", {})[s] = rv.rolling(60, min_periods=15).std()
for s, df in series.items():
    ar = df["ret"].abs()
    factors.setdefault("volcluster_60", {})[s] = ar.rolling(60, min_periods=40).corr(ar.shift(1))
for s, df in series.items():
    m = df["close"] / df["close"].shift(30) - 1.0
    v = df["close"].pct_change().rolling(60, min_periods=30).std()
    factors.setdefault("mom30_vol60", {})[s] = pd.Series(safe_div(m, v), index=df.index)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if not ics:
        print(fid, "NO IC DATES")
        continue
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
    reg250 = s["regime"].get("last250", {})
    results[fid] = {"ic": s["ic"], "icir": s["icir"], "hit": s["hit"],
                    "n": s["n_ic_dates"], "last250_ic": reg250.get("ic"),
                    "last250_icir": reg250.get("icir")}
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} n={s['n_ic_dates']:5d} "
          f"l250_ic={reg250.get('ic')} l250_icir={reg250.get('icir')}")

json.dump(results, open("scripts/miner_2_20321028_baseline_results.json", "w"), indent=1, default=str)
print("DONE")
