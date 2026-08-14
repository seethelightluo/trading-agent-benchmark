"""miner_2 2035-06-07 cycle screen: NEW factor candidates on the 15-instrument
cross-asset benchmark. Data through visible_through=2035-06-06 (no look-ahead;
macro series clipped to VISIBLE).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 at 10d horizon, >=8 valid assets per date.
Method: daily cross-sectional Spearman (rank-aligned Pearson on pct ranks).
Batch focus (distinct from 03-15 and 04-26 screens):
  cross-asset relative momentum (xau/spx/wti relative), vol-conditioned dynamics
  (zscore_20_120, vol_regime_ratio_20_60, drawdown_20_vol, price_pressure_5_60),
  macro-conditional momentum (vix_cond_mom_20), higher moments (kurtosis_20,
  updown_skew_20), beta dynamics (beta_trend_spx_60, alpha_60,
  downside_corr_spx_20), range recovery (recovery_ratio_20), vol-adj trend
  (trend_pos_20_60).
"""
import sys, json, time, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, to_grid, load_macro, VISIBLE,
    safe_div, cross_sectional_rank, summarize, turnover_10d_rank,
    coverage_stats, MIN_ASSETS, fwd_by_horizon_dict,
)

OUT = "scripts/miner_2_20350607_screen_results.json"
DAYS = 2400
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
        out[s] = pd.DataFrame({
            "close": close, "ret": ret,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float),
        })
    return out


def macro_clipped(name):
    m = load_macro(name)
    if m is None:
        return None
    return m[m.index <= VISIBLE]


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
print("assets with data:", sorted(series.keys()), "elapsed", round(time.time() - T0, 1), flush=True)

# forward returns by horizon (own calendar -> grid)
fwd_ranks = {}
for h in (1, 2, 3, 5, 10, 20):
    d = {s: df["close"].shift(-h) / df["close"] - 1.0 for s, df in series.items()}
    fwd_ranks[h] = cross_sectional_rank(to_grid(d))

dates = np.array(GRID)
spx = series["SPX"]["close"]
spxr = spx.pct_change()
xau = series["XAU"]["close"]
xaur = xau.pct_change()
wti = series["WTI"]["close"]
wtir = wti.pct_change()
vix = macro_clipped("VIX")
vixr = vix.pct_change() if vix is not None else None

factors = {}


def put(fid, s, vals):
    if isinstance(vals, np.ndarray):
        vals = pd.Series(vals, index=series[s].index)
    factors.setdefault(fid, {})[s] = vals


def roll_beta(x, y, w, minp):
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


def roll_corr(x, y, w, minp):
    return x.astype(float).rolling(w, min_periods=minp).corr(y.astype(float).reindex(x.index))


for s, df in series.items():
    close = df["close"]
    ret = df["ret"]
    vol20 = ret.rolling(20, min_periods=10).std()
    vol60 = ret.rolling(60, min_periods=30).std()
    ma20 = close.rolling(20, min_periods=10).mean()
    ma60 = close.rolling(60, min_periods=30).mean()
    r20 = close / close.shift(20) - 1.0
    r60 = close / close.shift(60) - 1.0
    r5 = close / close.shift(5) - 1.0

    # ---- N1 zscore_20_120: (close-ma20)/(std120) - deviation from 20d mean in 120d vol units
    sd120 = close.rolling(120, min_periods=60).std()
    put("zscore_20_120", s, safe_div(close - ma20, sd120))

    # ---- N2 vol_regime_ratio_20_60: vol20/vol60 - short/long vol term structure
    put("vol_regime_ratio_20_60", s, safe_div(vol20, vol60))

    # ---- N3 drawdown_20_vol: (close - max20)/(vol20*close) - vol-normalized drawdown depth
    mx20 = close.rolling(20, min_periods=10).max()
    put("drawdown_20_vol", s, safe_div(close - mx20, vol20 * close))

    # ---- N4 price_pressure_5_60: 5d return / 60d vol (short-term pressure, mean-reversion candidate)
    put("price_pressure_5_60", s, safe_div(r5, vol60))

    # ---- N5 recovery_ratio_20: (close-min20)/(max20-min20) - position within 20d range
    mn20 = close.rolling(20, min_periods=10).min()
    put("recovery_ratio_20", s, safe_div(close - mn20, mx20 - mn20))

    # ---- N6 trend_pos_20_60: (ma20-ma60)/(vol20*close) - vol-adjusted trend position
    put("trend_pos_20_60", s, safe_div(ma20 - ma60, vol20 * close))

    # ---- N7 kurtosis_20: excess kurtosis of 20d returns (tail risk)
    put("kurtosis_20", s, ret.rolling(20, min_periods=10).kurt())

    # ---- N8 updown_skew_20: skew(up days) - skew(down days) asymmetry
    up = ret.where(ret > 0)
    dn = ret.where(ret < 0)
    put("updown_skew_20", s, up.rolling(20, min_periods=10).skew() - dn.rolling(20, min_periods=10).skew())

    # ---- N9 beta_trend_spx_60: 60d SPX beta minus 120d SPX beta (beta drift)
    b60 = roll_beta(ret, spxr.reindex(df.index), 60, 30)
    b120 = roll_beta(ret, spxr.reindex(df.index), 120, 60)
    put("beta_trend_spx_60", s, b60 - b120)

    # ---- N10 alpha_60: 60d return - beta60 * SPX 60d return (idiosyncratic momentum)
    spx_r60 = spxr.rolling(60, min_periods=30).sum().reindex(df.index)
    put("alpha_60", s, r60 - b60 * spx_r60)

    # ---- N11 downside_corr_spx_20: correlation of asset ret with SPX on SPX-down days (20d)
    dn_spx = spxr.reindex(df.index) < 0
    c = ret.rolling(20, min_periods=10).corr(spxr.reindex(df.index))
    # conditional: corr only on days where SPX down -> use rolling corr on masked series
    ret_m = ret.where(dn_spx)
    spx_m = spxr.reindex(df.index).where(dn_spx)
    put("downside_corr_spx_20", s, ret_m.rolling(20, min_periods=6).corr(spx_m))

    # ---- N12 vix_cond_mom_20: 20d momentum gated by VIX regime (only when VIX below its 60d median)
    if vix is not None:
        vix_med = vix.rolling(60, min_periods=30).median().reindex(df.index)
        gate = (vix.reindex(df.index) < vix_med).astype(float)
        put("vix_cond_mom_20", s, r20 * gate)

    # ---- N13 xau_relative_20: 20d return - XAU 20d return (defensive rotation)
    xau_r20 = (xau / xau.shift(20) - 1.0).reindex(df.index)
    put("xau_relative_20", s, r20 - xau_r20)

    # ---- N14 spx_relative_20: 20d return - SPX 20d return (equity-relative strength)
    spx_r20 = (spx / spx.shift(20) - 1.0).reindex(df.index)
    put("spx_relative_20", s, r20 - spx_r20)

    # ---- N15 wti_relative_20: 20d return - WTI 20d return (energy-relative strength)
    wti_r20 = (wti / wti.shift(20) - 1.0).reindex(df.index)
    put("wti_relative_20", s, r20 - wti_r20)

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
