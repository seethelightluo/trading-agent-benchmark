"""miner_3 2035-04-26 cycle screen: NEW factor candidates on 15-instrument cross-asset
universe, data through 2035-04-25 (visible; macro clipped to VISIBLE to avoid look-ahead).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (10d horizon), >=8 valid assets per date.
Method: daily cross-sectional Spearman (row-Pearson on pct-ranks).
Distinct candidate set vs 04-12 screen (gap/range/vol-ratio/trend-accel/skew-change
batch already tested there): this batch = RSI, downside-vol asymmetry, bollinger
position, gap autocorr, 60d sharpe, trend R2, eff ratio 60, recovery position,
vol-adj short mom, crypto beta 60, us10y beta, commodity-relative momentum,
range compression, 20d skew, worst-day proximity.
"""
import sys, json, time, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, to_grid, load_macro, VISIBLE,
    safe_div, cross_sectional_rank, summarize, turnover_10d_rank,
    coverage_stats, MIN_ASSETS,
)

OUT = "scripts/miner_3_20350426_screen_results.json"
DAYS = 2100
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
        d = pd.DataFrame({
            "close": close, "ret": ret,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float),
        })
        out[s] = d
    return out


def macro_clipped(name):
    m = load_macro(name)
    if m is None:
        return None
    return m[m.index <= VISIBLE]


def roll_beta_vec(x, y, w, minp):
    """rolling beta of x on y (per-asset own calendar)."""
    xx = x.astype(float)
    yy = y.astype(float)
    valid = xx.notna() & yy.notna()
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


def roll_corr_vec(x, y, w, minp):
    xx = x.astype(float)
    yy = y.astype(float)
    return xx.rolling(w, min_periods=minp).corr(yy)


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

# forward returns by horizon (own calendar -> grid)
fwd_ranks = {}
for h in (1, 2, 3, 5, 10, 20):
    d = {s: df["close"].shift(-h) / df["close"] - 1.0 for s, df in series.items()}
    fwd_ranks[h] = cross_sectional_rank(to_grid(d))

dates = np.array(GRID)
spx = series["SPX"]["close"]
spxr = spx.pct_change()
btc = series["BTC"]["close"] if "BTC" in series else None
btcr = btc.pct_change() if btc is not None else None
wti = series["WTI"]["close"] if "WTI" in series else None
wtir = wti.pct_change() if wti is not None else None
us10y = series["US10Y"]["close"] if "US10Y" in series else None
us10yr = us10y.pct_change() if us10y is not None else None
dxy = macro_clipped("DXY")
vix = macro_clipped("VIX")

factors = {}


def put(fid, s, vals):
    factors.setdefault(fid, {})[s] = vals


# ---- N1 rsi_14: classic Relative Strength Index (14d) - mean-reversion candidate
for s, df in series.items():
    up = df["ret"].clip(lower=0)
    dn = (-df["ret"]).clip(lower=0)
    au = up.rolling(14, min_periods=7).mean()
    ad = dn.rolling(14, min_periods=7).mean()
    rs = safe_div(au, ad)
    put("rsi_14", s, 100.0 - 100.0 / (1.0 + rs))

# ---- N2 downvol_ratio_20: downside vol / upside vol (asymmetric risk)
for s, df in series.items():
    up = df["ret"].clip(lower=0)
    dn = df["ret"].clip(upper=0)
    vu = up.rolling(20, min_periods=10).std()
    vd = dn.rolling(20, min_periods=10).std()
    put("downvol_ratio_20", s, safe_div(vd, vu))

# ---- N3 boll_pos_20: (close - ma20) / (2*std20) - bollinger position
for s, df in series.items():
    ma = df["close"].rolling(20, min_periods=10).mean()
    sd = df["close"].rolling(20, min_periods=10).std()
    put("boll_pos_20", s, safe_div(df["close"] - ma, 2.0 * sd))

# ---- N4 gap_auto_20: lag-1 autocorrelation of overnight gaps over 20d
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    g1 = gap.shift(1)
    m = gap.rolling(20, min_periods=10).mean()
    sd = gap.rolling(20, min_periods=10).std()
    put("gap_auto_20", s, ((gap - m) * (g1 - m)).rolling(20, min_periods=10).mean() / (sd * sd))

# ---- N5 sharpe_60: 60d mean/std of daily returns (risk-adjusted momentum)
for s, df in series.items():
    mu = df["ret"].rolling(60, min_periods=30).mean()
    sd = df["ret"].rolling(60, min_periods=30).std()
    put("sharpe_60", s, safe_div(mu, sd))

# ---- N6 trend_r2_60: R2 of linear fit of close vs time over 60d (trend consistency)
for s, df in series.items():
    x = np.arange(60, dtype=float)
    xm = x - x.mean()
    xx = (xm * xm).sum()
    r2s = df["close"].rolling(60, min_periods=30).apply(
        lambda y: float(np.corrcoef(xm, y - np.nanmean(y))[0, 1] ** 2) if np.isfinite(y).sum() >= 30 else np.nan,
        raw=True)
    put("trend_r2_60", s, r2s)

# ---- N7 eff_ratio_60: |net move| / sum|ret| over 60d (trend efficiency)
for s, df in series.items():
    path = df["ret"].abs().rolling(60, min_periods=30).sum()
    move = (df["close"] / df["close"].shift(60) - 1.0).abs()
    put("eff_ratio_60", s, safe_div(move, path))

# ---- N8 recovery_pos_20_60: (close - min20) / (max60 - min60) - recovery position
for s, df in series.items():
    mn20 = df["close"].rolling(20, min_periods=10).min()
    mx60 = df["close"].rolling(60, min_periods=30).max()
    mn60 = df["close"].rolling(60, min_periods=30).min()
    put("recovery_pos_20_60", s, safe_div(df["close"] - mn20, mx60 - mn60))

# ---- N9 vol_adj_mom10_60: 10d return / 60d vol (short mom normalized)
for s, df in series.items():
    m10 = df["close"] / df["close"].shift(10) - 1.0
    v60 = df["ret"].rolling(60, min_periods=30).std()
    put("vol_adj_mom10_60", s, safe_div(m10, v60))

# ---- N10 crypto_beta_60: rolling beta vs BTC returns (crypto sensitivity)
if btcr is not None:
    for s, df in series.items():
        b = roll_beta_vec(df["ret"], btcr.reindex(df.index), 60, 30)
        put("crypto_beta_60", s, b)

# ---- N11 us10y_beta_60: rolling beta vs US10Y pct change (rates sensitivity)
if us10yr is not None:
    for s, df in series.items():
        b = roll_beta_vec(df["ret"], us10yr.reindex(df.index), 60, 30)
        put("us10y_beta_60", s, b)

# ---- N12 comm_rel_20: asset 20d return - WTI 20d return (commodity-relative strength)
if wti is not None:
    wtim = wti / wti.shift(20) - 1.0
    for s, df in series.items():
        m20 = df["close"] / df["close"].shift(20) - 1.0
        put("comm_rel_20", s, m20 - wtim.reindex(df.index))

# ---- N13 range_compress_10_60: 10d avg range / 60d avg range
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"]
    r10 = rng.rolling(10, min_periods=5).mean()
    r60 = rng.rolling(60, min_periods=30).mean()
    put("range_compress_10_60", s, safe_div(r10, r60))

# ---- N14 ret_skew_20: 20d return skewness (lottery/crash asymmetry)
for s, df in series.items():
    put("ret_skew_20", s, df["ret"].rolling(20, min_periods=10).skew())

# ---- N15 worst_day_10: min daily return over 10d (crash proximity, negative factor)
for s, df in series.items():
    put("worst_day_10", s, df["ret"].rolling(10, min_periods=5).min())

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
    s = summarize(ics, dates, fid, 10)
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
