"""
miner_3 cycle 2028-04-11: screen round 29 (data thru 2028-04-10).
Regime context (prior cycles): VIX re-escalating (19.98 -> 29.16 HIGH-vol),
XAU-led defensive floor, high dispersion, negative pairwise corr,
frozen feeds NDX/SOX/000688.SH/CN10Y (~27% dead weight).

Round-29 candidates (new formulations, avoiding previously-failed families):
  - gain_loss_ratio_20    : mean up-day ret / |mean down-day ret| (asymmetry)
  - overnight_ret_20      : 20d sum of overnight log ret (open/prev_close-1)
  - intraday_ret_20       : 20d sum of intraday log ret (close/open-1)
  - ma20_ma60_trend       : MA20/MA60 - 1 (classic smoothed trend)
  - up_vol_down_vol_20    : up-day vol / down-day vol (asymmetric vol)
  - cvar_20_neg           : negative 5% CVaR of 20d daily returns (tail risk)
  - volume_ret_corr_20    : corr(daily ret, volume pct change, 20d)
  - days_since_high_60_neg: negative days since 60d high (breakout recency)
  - dxy_trend_cond_mom    : 20d mom * sign(DXY 20d trend) (USD-conditional mom)
  - crypto_corr_20        : 20d rolling corr of asset ret with BTC ret
  - zscore_20d            : (close - MA20)/std20 (mean reversion)
  - range_vol_ratio_20    : mean(high-low)/close vs close-close vol (gap ratio)

Admission gates (warm-up 2020-01-01..2026-07-15): |IC_h10| >= 0.007,
|ICIR_h10| >= 0.084, max_abs_library_correlation < 0.5.
Live window 2026-07-16..2028-04-10 reported as informational drift.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS = 8
WARM_END = "2026-07-15"
LIVE_START = "2026-07-16"
DAYS = 4000
IC_GATE, ICIR_GATE, CORR_GATE = 0.007, 0.084, 0.5


def load_ohlcv(days=DAYS):
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=days)
        if df is None or not len(df):
            continue
        df = df.set_index("date")
        closes[s] = df["close"].astype(float)
        opens[s] = df["open"].astype(float)
        highs[s] = df["high"].astype(float)
        lows[s] = df["low"].astype(float)
        vols[s] = df["volume"].astype(float)

    def _p(d):
        p = pd.concat(d, axis=1, sort=True)
        return p[~p.index.duplicated(keep="last")].sort_index()
    return _p(closes), _p(opens), _p(highs), _p(lows), _p(vols)


def load_macro():
    out = {}
    for s in MACRO:
        df = get_index_daily_data(s, days=DAYS)
        if df is not None and len(df):
            out[s] = df.set_index("date")["close"].astype(float)
    return out


def per_asset(fn):
    def wrapper(panel):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fwd_returns(panel, h):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor, fwd):
    f = factor.stack().rename("f")
    r = fwd.stack().rename("r")
    j = pd.concat([f, r], axis=1).dropna()
    if len(j) == 0:
        return pd.Series(dtype=float)
    j["fr"] = j.groupby(level=0)["f"].rank()
    j["rr"] = j.groupby(level=0)["r"].rank()
    cnt = j.groupby(level=0).size()
    keep = cnt[cnt >= MIN_ASSETS].index
    j = j[j.index.get_level_values(0).isin(keep)]
    g = j.groupby(level=0)
    n = g.size()
    sx, sy = g["fr"].sum(), g["rr"].sum()
    sxx = g["fr"].apply(lambda s: float((s ** 2).sum()))
    syy = g["rr"].apply(lambda s: float((s ** 2).sum()))
    sxy = g.apply(lambda d: float((d["fr"] * d["rr"]).sum()))
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    ic = num / den
    return ic.sort_index()


def turnover_10d_rank(factor):
    ranks = factor.rank(axis=1)
    out, dates = [], ranks.index
    for i in range(10, len(dates)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) >= MIN_ASSETS:
            out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def stacked_corr(factor, libsig):
    out = {}
    f = factor.stack().rename("f")
    for fid, sig in libsig.items():
        s = sig.stack().rename("x")
        j = pd.concat([f, s], axis=1).dropna()
        if len(j) > 100:
            out[fid] = float(j["f"].corr(j["x"]))
    return out


def library_signals(close, high, low, vol, macro):
    lib = {}
    r = close.pct_change()
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    lib["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()

    def ew_beta(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = per_asset(ew_beta)(close)

    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(20).mean())
        tot = rr.rolling(20).std()
        return -(ds / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)(close)
    lib["max_ret_20d"] = r.rolling(20).max()

    def ew_corr(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).corr(z["m"])
    lib["corr_ew_60"] = per_asset(ew_corr)(close)

    def fx_cond(ref):
        ref20 = (ref / ref.shift(20) - 1.0)

        def f(s):
            z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
            return beta * ref20.reindex(s.index)
        return per_asset(f)(close)
    lib["dxy_beta_cond_60x20"] = fx_cond(macro["DXY"].dropna())
    lib["eurusd_beta_cond_60x20"] = fx_cond(macro["EURUSD"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-29 candidates ----------------
def cand_gain_loss_ratio_20(close):
    """Mean up-day ret / |mean down-day ret| over 20d."""
    def f(s):
        rr = s.pct_change()
        up = rr.where(rr > 0, np.nan).rolling(20, min_periods=8).mean()
        dn = rr.where(rr < 0, np.nan).rolling(20, min_periods=8).mean()
        return up / dn.abs()
    return per_asset(f)(close)


def cand_overnight_ret_20(close, open_):
    """20d rolling sum of overnight log ret (open/prev_close - 1)."""
    def f(s, o):
        gap = np.log(o / s.shift(1))
        return gap.rolling(20, min_periods=10).sum()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s, o = close[a].dropna(), open_[a].dropna()
        out[a] = f(s, o).reindex(close.index)
    return out


def cand_intraday_ret_20(close, open_):
    """20d rolling sum of intraday log ret (close/open - 1)."""
    def f(s, o):
        intr = np.log(s / o)
        return intr.rolling(20, min_periods=10).sum()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s, o = close[a].dropna(), open_[a].dropna()
        out[a] = f(s, o).reindex(close.index)
    return out


def cand_ma20_ma60_trend(close):
    """MA20/MA60 - 1 (smoothed trend position)."""
    def f(s):
        return s.rolling(20, min_periods=10).mean() / s.rolling(60, min_periods=30).mean() - 1.0
    return per_asset(f)(close)


def cand_up_vol_down_vol_20(close):
    """Up-day vol / down-day vol over 20d (asymmetric vol premium)."""
    def f(s):
        rr = s.pct_change()
        up = rr.where(rr > 0, np.nan).rolling(20, min_periods=8).std()
        dn = rr.where(rr < 0, np.nan).rolling(20, min_periods=8).std()
        return up / dn
    return per_asset(f)(close)


def cand_cvar_20_neg(close):
    """Negative 5% CVaR of daily returns over 20d (low tail-risk premium)."""
    def f(s):
        rr = s.pct_change()
        def _cvar(w):
            w = np.asarray(w, dtype=float)
            w = w[~np.isnan(w)]
            if len(w) < 5:
                return np.nan
            return float(np.mean(np.sort(w)[:max(1, int(np.ceil(0.05 * len(w))))]))
        return rr.rolling(20, min_periods=10).apply(_cvar, raw=True)
    return -per_asset(f)(close)


def cand_volume_ret_corr_20(close, vol):
    """Rolling corr(daily ret, volume pct change, 20d)."""
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s, v = close[a].dropna(), vol[a].dropna()
        r = s.pct_change()
        vc = v.pct_change()
        z = pd.concat([r.rename("r"), vc.rename("v")], axis=1).replace([np.inf, -np.inf], np.nan)
        out[a] = z["r"].rolling(20, min_periods=10).corr(z["v"]).reindex(close.index)
    return out


def cand_days_since_high_60_neg(close):
    """Negative days since 60d rolling high (breakout recency)."""
    def f(s):
        roll_max = s.rolling(60, min_periods=30).max()
        days = pd.Series(np.nan, index=s.index)
        vals, idx = s.values, s.index
        for i in range(30, len(vals)):
            w = vals[max(0, i - 59):i + 1]
            if np.isnan(w).any():
                continue
            mx = np.nanmax(w)
            if not np.isnan(mx) and mx > 0 and vals[i] >= mx * 0.9999:
                days.iloc[i] = 0
            else:
                prev = days.iloc[i - 1]
                days.iloc[i] = 0.0 if np.isnan(prev) else prev + 1.0
        return -days
    return per_asset(f)(close)


def cand_dxy_trend_cond_mom(close, macro):
    """20d momentum (skip5) * sign(DXY 20d trend)."""
    dxy = macro["DXY"].dropna()
    dxy_trend = np.sign(dxy / dxy.shift(20) - 1.0)
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        out[a] = (m20[a] * dxy_trend.reindex(s.index)).reindex(close.index)
    return out


def cand_crypto_corr_20(close):
    """20d rolling corr of asset daily ret with BTC daily ret (crypto linkage)."""
    btc = close["BTC"].dropna()
    br = btc.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), br.reindex(s.index).rename("b")], axis=1).dropna()
        out[a] = z["r"].rolling(20, min_periods=10).corr(z["b"]).reindex(close.index)
    return out


def cand_zscore_20d(close):
    """(close - MA20)/std20 (mean-reversion z-score, negated direction by IC)."""
    def f(s):
        return (s - s.rolling(20, min_periods=10).mean()) / s.rolling(20, min_periods=10).std()
    return per_asset(f)(close)


def cand_range_vol_ratio_20(close, high, low):
    """Mean (high-low)/close over 20d divided by close-close 20d vol (gap/intraday ratio)."""
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s, h, l = close[a].dropna(), high[a].dropna(), low[a].dropna()
        amp = ((h - l) / s).rolling(20, min_periods=10).mean()
        cv = s.pct_change().rolling(20, min_periods=10).std()
        out[a] = (amp / cv).reindex(close.index)
    return out


def validate(name, factor, close, libsig, window_end=WARM_END):
    res = {"n_dates": int(factor.loc[:window_end].shape[0])}
    fwd10 = fwd_returns(close, 10)
    ic = rank_ic_series(factor.loc[:window_end], fwd10)
    direction = 1.0 if ic.mean() >= 0 else -1.0
    res["ic_h10"] = float(direction * ic.mean())
    res["icir_h10"] = float(direction * ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic > 0).mean()) if len(ic) else float("nan")
    res["n_h10"] = len(ic)
    res["direction"] = direction
    res["decay"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        ic_h = rank_ic_series(factor.loc[:window_end], fwd_returns(close, h))
        res["decay"][str(h)] = float(direction * ic_h.mean()) if len(ic_h) else float("nan")
    valid = factor.loc[:window_end].notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(factor.loc[:window_end])
    corrs = stacked_corr(factor.loc[:window_end], libsig)
    res["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))}
    gate = abs(res["ic_h10"]) >= IC_GATE and abs(res["icir_h10"]) >= ICIR_GATE
    lowcorr = res["max_abs_library_correlation"] < CORR_GATE
    res["PASS"] = bool(gate and lowcorr)

    # live-window drift (informational)
    if factor.index[-1] > pd.Timestamp(LIVE_START):
        try:
            ic_l = rank_ic_series(factor.loc[LIVE_START:], fwd10)
            res["live_ic_h10"] = float(direction * ic_l.mean()) if len(ic_l) else float("nan")
            res["live_icir_h10"] = float(direction * ic_l.mean() / ic_l.std()) if len(ic_l) > 2 and ic_l.std() > 0 else float("nan")
            res["live_n"] = len(ic_l)
        except Exception:
            res["live_ic_h10"], res["live_icir_h10"], res["live_n"] = float("nan"), float("nan"), 0

    print(f"=== {name} === dates={res['n_dates']} direction={direction:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  decay={res['decay']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    if "live_ic_h10" in res:
        print(f"  LIVE {LIVE_START}..{factor.index[-1].date()}: IC={res['live_ic_h10']:+.4f} ICIR={res['live_icir_h10']:+.4f} n={res['live_n']}")
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}\n")

    print("regime sanity: ", end="")
    for s in ["VIX", "DXY", "EURUSD", "USDJPY", "USDCNY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%} ", end="")
    print("\n")

    cands = {
        "gain_loss_ratio_20": lambda: cand_gain_loss_ratio_20(close),
        "overnight_ret_20": lambda: cand_overnight_ret_20(close, open_),
        "intraday_ret_20": lambda: cand_intraday_ret_20(close, open_),
        "ma20_ma60_trend": lambda: cand_ma20_ma60_trend(close),
        "up_vol_down_vol_20": lambda: cand_up_vol_down_vol_20(close),
        "cvar_20_neg": lambda: cand_cvar_20_neg(close),
        "volume_ret_corr_20": lambda: cand_volume_ret_corr_20(close, vol),
        "days_since_high_60_neg": lambda: cand_days_since_high_60_neg(close),
        "dxy_trend_cond_mom": lambda: cand_dxy_trend_cond_mom(close, macro),
        "crypto_corr_20": lambda: cand_crypto_corr_20(close),
        "zscore_20d": lambda: cand_zscore_20d(close),
        "range_vol_ratio_20": lambda: cand_range_vol_ratio_20(close, high, low),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===\n")

    print("##### SUMMARY #####")
    for name, res_ in results.items():
        live = f" liveIC={res_['live_ic_h10']:+.4f}" if "live_ic_h10" in res_ else ""
        print(f"{name}: IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} hit={res_['hit_h10']:.3f} "
              f"cov_ge8={res_['coverage_dates_ge8']:.3f} turn={res_['turnover_10d_rank']:.3f} "
              f"maxcorr={res_['max_abs_library_correlation']:.3f}{live} "
              f"-> {'PASS' if res_['PASS'] else 'FAIL'}")
