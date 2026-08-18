"""
miner_3 cycle 2027-06-08: screen round 25 (data thru 2027-06-07).
Re-validates the 7-factor library and screens round-25 candidates for the
current regime:
  VIX ~21+ elevated (2026 9->47 precedent), post 05-25->06-08 block where
  CN10Y -4.71% (3rd bond whipsaw), COPPER -8.1%, ETH -12.2% reversals while
  BTC +6.7% / WTI +5.4% bounced (rel_mom whipsaw), SPX +4.25% resilient,
  XAU flat +0.4%, frozen feeds 000688.SH/SOX/NDX.

Theme of round 25: stress-regime robustness.
  - trend-confirmed momentum (address rel_mom whipsaw)
  - drawdown time/depth (oversold timing)
  - asymmetric down/up beta vs EW (tail protection, complements downside_vol)
  - skewness (complements kurt), risk-adjusted momentum (Sharpe-like)
  - overnight gap / flow (OBV slope) / stochastic position (mean reversion)
  - cross-asset affinities (BTC, WTI-beta, US10Y) and trend efficiency

Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084,
max_abs_library_correlation < 0.5. Validation window warm-up thru 2026-07-15;
live drift 2026-07-16..end informational.
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


# ---------------- current library signals (7 kept) ----------------
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
    lib["eurusd_beta_cond_60x20"] = fx_cond(macro["EURUSD"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-25 new candidates ----------------
def cand_trend_mom_20x60(close):
    # trend-confirmed momentum: 20d mom(skip5) * trend position (close/MA60 - 1)
    mom20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    ma60 = close.rolling(60, min_periods=30).mean()
    trend = close / ma60 - 1.0
    out = mom20 * trend
    return out.sub(out.median(axis=1), axis=0)


def cand_dd_time_60(close):
    # -days since 60d high (drawdown duration)
    def dtime(s, n=60):
        arr = s.values
        out = np.full(len(s), np.nan)
        for i in range(n - 1, len(s)):
            w = arr[i - n + 1:i + 1]
            if np.isnan(w).any():
                continue
            hi = np.nanmax(w)
            idx = np.where(w == hi)[0][-1]
            out[i] = -(len(w) - 1 - idx)
        return pd.Series(out, index=s.index)
    return per_asset(dtime)(close)


def cand_maxdd_60(close):
    m60 = close.rolling(60, min_periods=30).max()
    return close / m60 - 1.0


def cand_asym_beta_60(close, ew_r):
    # down-market beta - up-market beta (vs EW index); higher = tail-protective
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), ew_r.rename("m")], axis=1).dropna()
        up = z["r"].where(z["m"] > 0)
        dn = z["r"].where(z["m"] <= 0)
        m_up = z["m"].where(z["m"] > 0)
        m_dn = z["m"].where(z["m"] <= 0)
        b_up = up.rolling(60, min_periods=20).cov(m_up) / m_up.rolling(60, min_periods=20).var()
        b_dn = dn.rolling(60, min_periods=20).cov(m_dn) / m_dn.rolling(60, min_periods=20).var()
        return (b_dn - b_up).reindex(s.index)
    return per_asset(f)(close)


def cand_skew_20d_skip5(close):
    r = close.pct_change().shift(5)
    return r.rolling(20, min_periods=12).skew()


def cand_sharpe_20x60(close):
    r = close.pct_change()
    mu20 = r.rolling(20, min_periods=10).mean()
    sd60 = r.rolling(60, min_periods=30).std()
    return mu20 / (sd60 + 1e-9)


def cand_gap_20(close, open_):
    gap = open_ / close.shift(1) - 1.0
    return gap.rolling(20, min_periods=10).mean()


def cand_stoch_20(close):
    lo20 = close.rolling(20, min_periods=10).min()
    hi20 = close.rolling(20, min_periods=10).max()
    return (close - lo20) / (hi20 - lo20).replace(0, np.nan)


def cand_obv_slope_20(close, vol):
    r = close.pct_change()
    sign = np.sign(r)
    obv = (sign * vol).cumsum()
    slope = obv - obv.shift(20)
    vmean = vol.rolling(20, min_periods=10).mean()
    return slope / (vmean * 20 + 1e-9)


def cand_btc_affinity_60(close):
    r = close.pct_change()
    btc_r = r["BTC"]
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = r[a].rolling(60, min_periods=30).corr(btc_r)
    return out


def cand_wti_beta_cond_60x20(close):
    wti = close["WTI"].dropna()
    wti_r = wti.pct_change()
    wti20 = (wti / wti.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), wti_r.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        out[a] = (beta * wti20.reindex(s.index)).reindex(close.index)
    return out


def cand_us10y_affinity_60(close):
    r = close.pct_change()
    u10 = r["US10Y"]
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = r[a].rolling(60, min_periods=30).corr(u10)
    return out


def cand_eff_ratio_10x60(close):
    def eff(s, n=10):
        net = (s / s.shift(n) - 1.0).abs()
        path = s.pct_change().abs().rolling(n, min_periods=n).sum()
        return net / (path + 1e-9)
    return per_asset(eff)(close)


def cand_realized_range_20(close, high, low):
    rr = (high - low) / close
    return rr.rolling(20, min_periods=10).mean()


def cand_rel_rev_5d(close):
    r5 = per_asset(lambda s: s / s.shift(5) - 1.0)(close)
    rel = r5.sub(r5.median(axis=1), axis=0)
    return -rel


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
    print(f"=== {name} === dates={res['n_dates']} direction={direction:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  decay={res['decay']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}\n")

    r = close.pct_change()
    print("regime sanity: ", end="")
    for s in ["VIX", "DXY", "EURUSD", "USDJPY", "USDCNY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%} ", end="")
    print()
    for s in WATCH:
        if s in close.columns:
            print(f"{s} 20d={r[s].iloc[-20:].add(1).prod()-1:+.1%} 60d={r[s].iloc[-60:].add(1).prod()-1:+.1%}")
    mcorr = r.iloc[-60:].corr().abs().stack()
    mcorr = mcorr[mcorr < 0.999]
    print(f"mean |pairwise corr| last 60d: {mcorr.mean():.4f}")
    disp = r.sub(r.mean(axis=1), axis=0).abs().mean(axis=1)
    print(f"cross-sectional daily dispersion last 20d mean: {disp.iloc[-20:].mean()*100:.2f}%")
    print()

    print("##### A1) LIBRARY WARM-UP VALIDATION (thru 2026-07-15) #####")
    for fid, sig in libsig.items():
        validate(fid, sig, close, libsig)

    print("##### A2) LIBRARY LIVE-WINDOW DRIFT (informational: 2026-07-16..end) #####")
    for fid, sig in libsig.items():
        sub = sig.loc[LIVE_START:]
        if sub.notna().sum().sum() < 200:
            print(f"  {fid}: too few live obs, skip")
            continue
        ic = rank_ic_series(sub, fwd_returns(close, 10).loc[LIVE_START:])
        if len(ic) < 5:
            print(f"  {fid}: live IC n={len(ic)} too small")
            continue
        d = 1.0 if ic.mean() >= 0 else -1.0
        icir = d * ic.mean() / ic.std() if ic.std() > 0 else float("nan")
        print(f"  {fid}: live h10 IC={d*ic.mean():+.4f} ICIR={icir:+.4f} hit={(d*ic>0).mean():.3f} n={len(ic)}")
    print()

    print("##### B) ROUND-25 CANDIDATE SCREENS #####")
    ew_r = close.mean(axis=1).pct_change()
    cands = {
        "trend_mom_20x60": lambda: cand_trend_mom_20x60(close),
        "dd_time_60": lambda: cand_dd_time_60(close),
        "maxdd_60": lambda: cand_maxdd_60(close),
        "asym_beta_60": lambda: cand_asym_beta_60(close, ew_r),
        "skew_20d_skip5": lambda: cand_skew_20d_skip5(close),
        "sharpe_20x60": lambda: cand_sharpe_20x60(close),
        "gap_20": lambda: cand_gap_20(close, open_),
        "stoch_20": lambda: cand_stoch_20(close),
        "obv_slope_20": lambda: cand_obv_slope_20(close, vol),
        "btc_affinity_60": lambda: cand_btc_affinity_60(close),
        "wti_beta_cond_60x20": lambda: cand_wti_beta_cond_60x20(close),
        "us10y_affinity_60": lambda: cand_us10y_affinity_60(close),
        "eff_ratio_10x60": lambda: cand_eff_ratio_10x60(close),
        "realized_range_20": lambda: cand_realized_range_20(close, high, low),
        "rel_rev_5d": lambda: cand_rel_rev_5d(close),
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
        print(f"{name:<22} IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} maxcorr={res_['max_abs_library_correlation']:.3f} cov_ge8={res_['coverage_dates_ge8']:.2f} turn={res_['turnover_10d_rank']:.2f} -> {'PASS' if res_['PASS'] else 'FAIL'}")
