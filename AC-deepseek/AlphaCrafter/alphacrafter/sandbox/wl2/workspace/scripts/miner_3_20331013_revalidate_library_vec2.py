"""miner_3 2033-10-13: full library revalidation v2 - fully vectorized IC machinery
(data through 2033-10-12 visible). Fixes days_since_high_60 (inverted where mask) and
replaces per-date pandas rank-corr loops with vectorized row-Pearson on pre-ranked matrices
(identical math: Spearman = Pearson on pct-ranks with pairwise deletion).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (15-instrument cross-asset universe, >=8 valid per date).
"""
import sys, json, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, to_grid, load_macro,
    safe_div, cross_sectional_rank,
    summarize, turnover_10d_rank, coverage_stats, HORIZON, MIN_ASSETS,
)

OUT = "scripts/miner_3_20331013_revalidate_results.json"
DAYS = 4200
T0 = time.time()


def load_asset_raw(sym, days=DAYS):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset_raw(s)
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


def roll_beta_cond_vec(x, y, w, minp, cond=None):
    """beta = cov(x,y|valid)/var(x|valid) over trailing w-window, min minp valid obs.
    Vectorized via rolling sums (exact equivalent of per-day np.cov loop)."""
    xx = x.astype(float)
    yy = y.astype(float)
    valid = xx.notna() & yy.notna()
    if cond is not None:
        valid = valid & cond.astype(bool)
    m = valid.astype(float)
    sx = (xx * m).rolling(w, min_periods=minp).sum()
    sy = (yy * m).rolling(w, min_periods=minp).sum()
    sxy = (xx * yy * m).rolling(w, min_periods=minp).sum()
    sxx = (xx * xx * m).rolling(w, min_periods=minp).sum()
    n = m.rolling(w, min_periods=minp).sum()
    mx = sx / n
    my = sy / n
    cov = sxy / n - mx * my
    varx = sxx / n - mx * mx
    beta = cov / varx
    return beta.where(varx > 1e-12)


def streak_count(pos):
    """Consecutive-run count: 0 where pos==0; 1,2,3.. on pos==1 runs (exact vs loop)."""
    cs = pos.cumsum()
    last_reset = cs.where(pos == 0).ffill().fillna(0.0)
    return (cs - last_reset).where(pos == 1, 0.0)


def days_since_flag(flag):
    """Days since last flag==1: high day -> 0; non-high days count 1,2,3.. since last high."""
    not_flag = (flag == 0).astype(float)
    cs = not_flag.cumsum()
    last_reset = cs.where(flag == 1).ffill().fillna(0.0)
    return (cs - last_reset).where(flag == 0, 0.0)


def row_pearson(X, Y, minp=MIN_ASSETS):
    """Row-wise Pearson with pairwise deletion; NaN where < minp valid pairs.
    Spearman = Pearson on pct-ranks, so ICs are rank ICs when inputs are rank matrices."""
    V = ~(np.isnan(X) | np.isnan(Y))
    Xm = np.where(V, X, np.nan)
    Ym = np.where(V, Y, np.nan)
    mx = np.nanmean(Xm, axis=1, keepdims=True)
    my = np.nanmean(Ym, axis=1, keepdims=True)
    dx = Xm - mx
    dy = Ym - my
    num = np.nansum(dx * dy, axis=1)
    den = np.sqrt(np.nansum(dx * dx, axis=1) * np.nansum(dy * dy, axis=1))
    ic = num / den
    ic[np.sum(V, axis=1) < minp] = np.nan
    return ic


def library_pairwise_corr_ex(factor_mat, fid):
    """Spearman rho vs every factors/*.signal.npy artifact except the factor's own."""
    import glob, os
    ours = cross_sectional_rank(factor_mat)
    out = {}
    for f in sorted(glob.glob("factors/*.signal.npy")):
        name = os.path.basename(f).replace(".signal.npy", "")
        if name == fid:
            continue
        arr = np.load(f, allow_pickle=True)
        rows = min(arr.shape[0], ours.shape[0])
        a = ours[:rows]
        b = arr[:rows]
        rho = None
        for t in range(rows):
            x = a[t]; y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= MIN_ASSETS:
                xs = pd.Series(x[ok]).rank(); ys = pd.Series(y[ok]).rank()
                c = xs.corr(ys)
                if np.isfinite(c):
                    rho = c
                    break
        if rho is not None:
            out[name] = round(float(rho), 4)
    if out:
        mx = max(out.items(), key=lambda kv: abs(kv[1]))
        return out, mx[0], abs(mx[1])
    return out, None, 0.0


series = asset_series_full()
print("assets with data:", sorted(series.keys()), "elapsed", round(time.time() - T0, 1))

# forward-return rank matrices for all horizons (precomputed once)
fwd_ranks = {}
for h in (1, 2, 3, 5, 10, 20):
    d = {s: df["close"].shift(-h) / df["close"] - 1.0 for s, df in series.items()}
    fwd_ranks[h] = cross_sectional_rank(to_grid(d))

dates = np.array(GRID)
spx = series["SPX"]["close"]
spxr = spx.pct_change()
dxy = load_macro("DXY")
usdjpy = load_macro("USDJPY")
vix = load_macro("VIX")

factors = {}

# 1 calmness_20
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"]
    factors.setdefault("calmness_20", {})[s] = 1.0 - rng.rolling(20, min_periods=10).mean()

# 2 close_pos_20
for s, df in series.items():
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = (df["close"] - df["low"]) / rng
    factors.setdefault("close_pos_20", {})[s] = pos.rolling(20, min_periods=10).mean()

# 3 days_since_high_60
for s, df in series.items():
    rl = df["close"].rolling(60, min_periods=30).max()
    dsh = (df["close"] == rl).astype(float)
    out = days_since_flag(dsh)
    factors.setdefault("days_since_high_60", {})[s] = out.rolling(20, min_periods=10).max()

# 4 downbeta_spx_60
for s, df in series.items():
    spxr_s = spxr.reindex(df.index)
    factors.setdefault("downbeta_spx_60", {})[s] = roll_beta_cond_vec(df["ret"], spxr_s, 60, 30, cond=spxr_s < 0)

# 5 dxy_beta_cond_60x20
for s, df in series.items():
    if dxy is None:
        continue
    dy = dxy.reindex(df.index)
    beta = roll_beta_cond_vec(df["ret"], dy.pct_change(), 60, 30)
    m = dy / dy.shift(20) - 1.0
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = beta * m

# 6 gain_loss_20
for s, df in series.items():
    g = df["ret"].clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-df["ret"].clip(upper=0)).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = pd.Series(safe_div(g - l, g + l), index=df.index)

# 7 intraday_drift_20
for s, df in series.items():
    drift = df["close"] / df["open"] - 1.0
    factors.setdefault("intraday_drift_20", {})[s] = drift.rolling(20, min_periods=10).mean()

# 8 lagbeta_spx_60
for s, df in series.items():
    factors.setdefault("lagbeta_spx_60", {})[s] = roll_beta_cond_vec(df["ret"], spxr.shift(1).reindex(df.index), 60, 30)

# 9 max_consec_gain_20
for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    factors.setdefault("max_consec_gain_20", {})[s] = streak_count(pos).rolling(20, min_periods=10).max()

# 10 max_consec_loss_20
for s, df in series.items():
    neg = (df["ret"] < 0).astype(float)
    factors.setdefault("max_consec_loss_20", {})[s] = streak_count(neg).rolling(20, min_periods=10).max()

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
    lo = df["close"].rolling(252, min_periods=120).min()
    hi = df["close"].rolling(252, min_periods=120).max()
    factors.setdefault("range_pos_252", {})[s] = pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)

# 17 spx_corr60
for s, df in series.items():
    a = df["ret"]
    b = spxr.reindex(df.index)
    factors.setdefault("spx_corr60", {})[s] = a.rolling(60, min_periods=30).corr(b)

# 18 usdjpy_beta_cond_120x60
for s, df in series.items():
    if usdjpy is None:
        continue
    uy = usdjpy.reindex(df.index)
    beta = roll_beta_cond_vec(df["ret"], uy.pct_change(), 120, 60)
    m = uy / uy.shift(60) - 1.0
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = beta * m

# 19 vix_beta_cond_60x20
for s, df in series.items():
    if vix is None:
        continue
    vx = vix.reindex(df.index)
    beta = roll_beta_cond_vec(df["ret"], vx.pct_change(), 60, 30)
    m = vx / vx.shift(20) - 1.0
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

print("all factor panels computed, elapsed", round(time.time() - T0, 1), flush=True)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ic10 = row_pearson(rank_mat, fwd_ranks[10])
    idx = np.where(np.isfinite(ic10))[0]
    if len(idx) == 0:
        print(fid, "NO IC DATES")
        continue
    ics = [(int(t), float(ic10[t])) for t in idx]
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
    decay = {}
    for h, fwd_r in fwd_ranks.items():
        ih = row_pearson(rank_mat, fwd_r)
        if np.any(np.isfinite(ih)):
            decay[str(h)] = round(float(np.nanmean(ih)), 4)
    s["decay"] = decay
    rho_dict, rho_name, max_rho = library_pairwise_corr_ex(mat, fid)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    reg250 = s["regime"].get("last250", {})
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT, "elapsed", round(time.time() - T0, 1))
