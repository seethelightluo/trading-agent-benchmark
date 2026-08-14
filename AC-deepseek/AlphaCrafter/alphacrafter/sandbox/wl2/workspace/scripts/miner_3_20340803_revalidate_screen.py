"""miner_3 2034-08-03: full library revalidation + new factor screen.
Data through 2034-08-02 (visible). Vectorized row-Pearson on pct-ranked matrices
(Spearman rank IC). Gates: |IC|>=0.0070, |ICIR|>=0.0840 (15-instrument cross-asset
universe, >=8 valid per date). Horizon 10d.
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

OUT = "scripts/miner_3_20340803_revalidate_results.json"
DAYS = 4300
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
    cs = pos.cumsum()
    last_reset = cs.where(pos == 0).ffill().fillna(0.0)
    return (cs - last_reset).where(pos == 1, 0.0)


def days_since_flag(flag):
    not_flag = (flag == 0).astype(float)
    cs = not_flag.cumsum()
    last_reset = cs.where(flag == 1).ffill().fillna(0.0)
    return (cs - last_reset).where(flag == 0, 0.0)


def row_pearson(X, Y, minp=MIN_ASSETS):
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
us10y = series["US10Y"]["close"] if "US10Y" in series else None
btc = series["BTC"]["close"] if "BTC" in series else None

factors = {}

# ============ EXISTING LIBRARY (revalidate) ============
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
    factors.setdefault("days_since_high_60", {})[s] = days_since_flag(dsh).rolling(20, min_periods=10).max()

for s, df in series.items():
    spxr_s = spxr.reindex(df.index)
    factors.setdefault("downbeta_spx_60", {})[s] = roll_beta_cond_vec(df["ret"], spxr_s, 60, 30, cond=spxr_s < 0)

for s, df in series.items():
    if dxy is None:
        continue
    dy = dxy.reindex(df.index)
    beta = roll_beta_cond_vec(df["ret"], dy.pct_change(), 60, 30)
    m = dy / dy.shift(20) - 1.0
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = beta * m

for s, df in series.items():
    g = df["ret"].clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-df["ret"].clip(upper=0)).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = pd.Series(safe_div(g - l, g + l), index=df.index)

for s, df in series.items():
    drift = df["close"] / df["open"] - 1.0
    factors.setdefault("intraday_drift_20", {})[s] = drift.rolling(20, min_periods=10).mean()

for s, df in series.items():
    factors.setdefault("lagbeta_spx_60", {})[s] = roll_beta_cond_vec(df["ret"], spxr.shift(1).reindex(df.index), 60, 30)

for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    factors.setdefault("max_consec_gain_20", {})[s] = streak_count(pos).rolling(20, min_periods=10).max()

for s, df in series.items():
    neg = (df["ret"] < 0).astype(float)
    factors.setdefault("max_consec_loss_20", {})[s] = streak_count(neg).rolling(20, min_periods=10).max()

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
    a = df["ret"]
    b = spxr.reindex(df.index)
    factors.setdefault("spx_corr60", {})[s] = a.rolling(60, min_periods=30).corr(b)

for s, df in series.items():
    if usdjpy is None:
        continue
    uy = usdjpy.reindex(df.index)
    beta = roll_beta_cond_vec(df["ret"], uy.pct_change(), 120, 60)
    m = uy / uy.shift(60) - 1.0
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = beta * m

for s, df in series.items():
    if vix is None:
        continue
    vx = vix.reindex(df.index)
    beta = roll_beta_cond_vec(df["ret"], vx.pct_change(), 60, 30)
    m = vx / vx.shift(20) - 1.0
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

# ============ NEW CANDIDATES ============
# N1 dd_depth_60: 60d drawdown depth (close vs trailing max), <=0; near-0 = shallow dd
for s, df in series.items():
    rl = df["close"].rolling(60, min_periods=30).max()
    factors.setdefault("dd_depth_60", {})[s] = df["close"] / rl - 1.0

# N2 rel_strength_spx_60: excess 60d return vs SPX
for s, df in series.items():
    m = df["close"] / df["close"].shift(60) - 1.0
    mspx = (spx / spx.shift(60) - 1.0).reindex(df.index)
    factors.setdefault("rel_strength_spx_60", {})[s] = m - mspx

# N3 vol_ratio_10_60: short/long vol term structure
for s, df in series.items():
    v10 = df["ret"].rolling(10, min_periods=5).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("vol_ratio_10_60", {})[s] = pd.Series(safe_div(v10, v60), index=df.index)

# N4 ret_skew_60: rolling skewness of 60d returns
for s, df in series.items():
    factors.setdefault("ret_skew_60", {})[s] = df["ret"].rolling(60, min_periods=30).skew()

# N5 tr_20: mean true range 20d / close
for s, df in series.items():
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    factors.setdefault("tr_20", {})[s] = pd.Series(safe_div(tr.rolling(20, min_periods=10).mean(), df["close"]), index=df.index)

# N6 max_gain_loss_ratio_20: max daily gain / |min daily loss| over 20d
for s, df in series.items():
    mg = df["ret"].rolling(20, min_periods=10).max()
    ml = (-df["ret"]).rolling(20, min_periods=10).max()
    factors.setdefault("max_gain_loss_ratio_20", {})[s] = pd.Series(safe_div(mg, ml), index=df.index)

# N7 mom_60d_skip5_volscale: 60d momentum skip5 scaled by vol60
for s, df in series.items():
    m = df["close"] / df["close"].shift(65) - 1.0
    v = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("mom_60d_skip5_volscale", {})[s] = pd.Series(safe_div(m, v), index=df.index)

# N8 btc_corr60: 60d return correlation with BTC
if btc is not None:
    btc_r = btc.pct_change()
    for s, df in series.items():
        factors.setdefault("btc_corr60", {})[s] = df["ret"].rolling(60, min_periods=30).corr(btc_r.reindex(df.index))

# N9 hl_range_pos_20: position within 20d high-low range (use high/low not close)
for s, df in series.items():
    lo = df["low"].rolling(20, min_periods=10).min()
    hi = df["high"].rolling(20, min_periods=10).max()
    factors.setdefault("hl_range_pos_20", {})[s] = pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)

# N10 eff_ratio_60: trend efficiency |move| / path length over 60d
for s, df in series.items():
    path = df["ret"].abs().rolling(60, min_periods=30).sum()
    move = (df["close"] / df["close"].shift(60) - 1.0).abs()
    factors.setdefault("eff_ratio_60", {})[s] = pd.Series(safe_div(move, path), index=df.index)

# N11 dd_depth_20: 20d drawdown depth
for s, df in series.items():
    rl = df["close"].rolling(20, min_periods=10).max()
    factors.setdefault("dd_depth_20", {})[s] = df["close"] / rl - 1.0

# N12 vol_zscore_60: current 20d vol relative to 120d vol level (z)
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v120 = df["ret"].rolling(120, min_periods=60).std()
    factors.setdefault("vol_zscore_60", {})[s] = pd.Series(safe_div(v20 - v120, v120), index=df.index)

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
