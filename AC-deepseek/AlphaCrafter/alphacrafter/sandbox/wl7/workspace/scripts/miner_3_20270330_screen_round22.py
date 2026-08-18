"""
miner_3 cycle 2027-03-30: screen round 22 (data thru 2027-03-29).
Re-validates the 8-factor library and re-screens round-21 candidates with
fresh data; adds round-22 candidates designed for the current regime:
VIX 9.0 pinned complacency, SPX +8.5%/20d rebound, BTC -7.5%/20d & WTI -7.8%/20d
downtrends, XAU -4.6%/20d (safe-haven gave back gains), CN10Y +6.8%/20d rally,
DXY +3.1%/20d USD strength, N225 -5.8%/20d (post-reversal weak).

Admission gates (same as prior cycles): |IC_h10| >= 0.007, |ICIR_h10| >= 0.084,
max_abs_library_correlation < 0.5. Validation window warm-up through 2026-07-15
plus live drift 2026-07-16..2027-03-29 (informational).

Round-22 new candidates:
  gap_ret_20            - signed cumulative overnight-gap return over 20d
  close_loc_20          - mean (close-low)/(high-low) over 20d (buying pressure)
  sharpe_60d            - 60d mean daily ret / std (risk-adjusted momentum)
  dd_recovery_20        - close/rolling_max(close,60)-1 (recovery vs 60d high)
  us10y_rs_20           - 20d asset ret minus 20d US10Y ret (bond-relative strength)
  xau_corr_60           - 60d correlation to XAU (safe-haven affinity)
  range_position_5      - (close-5d low)/(5d high-5d low) short-term breakout pos
  intraday_vol_share_20 - mean((high-low)/close)/20d ret std (intraday share)
  vol_beta_60           - 60d beta of asset 20d vol to EW 20d vol (vol sensitivity)
  mom_consistency_60    - 60d fraction of positive daily returns
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
    sxx, syy = (g["fr"] ** 2).sum(), (g["rr"] ** 2).sum()
    sxy = (g["fr"] * g["rr"]).sum()
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


# ---------------- current library signals (8 kept) ----------------
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


# ---------------- round-21 candidates (refresh) ----------------
def cand_trend_gated_mom_20(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    ma20 = close.rolling(20, min_periods=10).mean()
    gate = (close > ma20).astype(float)
    return (m20 - m20.median(axis=1)) * gate


def cand_smooth_mom_20(close):
    r = close.pct_change()
    ema = r.ewm(span=3, min_periods=3).mean()
    return ema.rolling(20, min_periods=10).sum()


def cand_zscore_20d(close):
    ma20 = close.rolling(20, min_periods=10).mean()
    sd20 = close.rolling(20, min_periods=10).std()
    return (close - ma20) / (sd20 + 1e-9)


def cand_breakout_20d(close):
    hi20 = close.rolling(20, min_periods=10).max()
    return close / hi20 - 1.0


def cand_updown_vol_asym_20(close):
    r = close.pct_change()

    def asym(w):
        w = np.asarray(w, dtype=float)
        up = w[w > 0]
        dn = w[w < 0]
        if len(up) < 2 or len(dn) < 2:
            return np.nan
        return float(up.std(ddof=1) / (dn.std(ddof=1) + 1e-9))
    return r.rolling(20, min_periods=12).apply(asym, raw=True)


def cand_roll_vol_ratio_20x60(close):
    r = close.pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    v60 = r.rolling(60, min_periods=30).std()
    return v20 / (v60 + 1e-9)


def cand_cross_dispersion_60(close):
    r = close.pct_change()
    cmean = r.mean(axis=1)
    dev = r.sub(cmean, axis=0).abs()
    return dev.rolling(60, min_periods=30).mean()


def cand_xau_beta_cond_20(close):
    xau = close["XAU"].dropna()
    xau20 = (xau / xau.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), xau.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        out[a] = (beta * xau20.reindex(s.index)).reindex(close.index)
    return out


def cand_us10y_mom_link_20(close, macro):
    us10y = macro["US10Y"].dropna()
    us10y20 = (us10y / us10y.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), us10y.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        out[a] = (beta * us10y20.reindex(s.index)).reindex(close.index)
    return out


def cand_wti_link_20(close):
    wti = close["WTI"].dropna()
    wti20 = (wti / wti.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), wti.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        out[a] = (beta * wti20.reindex(s.index)).reindex(close.index)
    return out


def cand_pos_return_ratio_20(close):
    r = close.pct_change()
    return (r > 0).rolling(20, min_periods=12).mean()


def cand_vol_gated_beta_60(close):
    r = close.pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    v60 = r.rolling(60, min_periods=30).std()
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()

    def ew_beta(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    beta = per_asset(ew_beta)(close)
    gate = (v20 < v60).astype(float)
    return beta * gate


# ---------------- round-22 new candidates ----------------
def cand_gap_ret_20(close, open_):
    gap = open_ / close.shift(1) - 1.0
    return gap.rolling(20, min_periods=12).sum()


def cand_close_loc_20(close, high, low):
    rng = (high - low).replace(0, np.nan)
    loc = (close - low) / rng
    return loc.rolling(20, min_periods=12).mean()


def cand_sharpe_60d(close):
    def sh(s):
        rr = s.pct_change()
        mu = rr.rolling(60, min_periods=30).mean()
        sd = rr.rolling(60, min_periods=30).std()
        return mu / (sd + 1e-9)
    return per_asset(sh)(close)


def cand_dd_recovery_20(close):
    max60 = close.rolling(60, min_periods=30).max()
    return close / max60 - 1.0


def cand_us10y_rs_20(close, macro):
    us10y = macro["US10Y"].dropna()
    us10y20 = (us10y / us10y.shift(20) - 1.0).reindex(close.index).ffill()
    m20 = close / close.shift(20) - 1.0
    return m20 - us10y20


def cand_xau_corr_60(close):
    r = close.pct_change()
    xau_r = r["XAU"]
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = r[a].rolling(60, min_periods=30).corr(xau_r)
    return out


def cand_range_position_5(close):
    lo = close.rolling(5, min_periods=4).min()
    hi = close.rolling(5, min_periods=4).max()
    return (close - lo) / (hi - lo).replace(0, np.nan)


def cand_intraday_vol_share_20(close, high, low):
    r = close.pct_change()
    intr = ((high - low) / close).rolling(20, min_periods=12).mean()
    tot = r.rolling(20, min_periods=12).std()
    return intr / (tot + 1e-9)


def cand_vol_beta_60(close):
    r = close.pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    ew_v = v20.mean(axis=1)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        z = pd.concat([v20[a].rename("v"), ew_v.rename("m")], axis=1).dropna()
        beta = z["v"].rolling(60, min_periods=30).cov(z["m"]) / z["m"].rolling(60, min_periods=30).var()
        out[a] = beta.reindex(close.index)
    return out


def cand_mom_consistency_60(close):
    r = close.pct_change()
    return (r > 0).rolling(60, min_periods=30).mean()


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
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}; visible_through expected 2027-03-29")
    print(f"library factors: {list(libsig.keys())}\n")

    r = close.pct_change()
    print("regime sanity: ", end="")
    for s in ["VIX", "DXY", "EURUSD", "USDJPY", "USDCNY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%} ", end="")
    print()
    for s in ["US10Y", "CN10Y", "XAU", "BTC", "ETH", "WTI", "COPPER", "SPX", "NDX", "SOX", "000300.SH", "000688.SH", "N225", "HSI", "SX5E"]:
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

    print("##### B) CANDIDATE SCREENS (round 21 refresh + round 22 new) #####")
    cands = {
        # round-21 refresh
        "trend_gated_mom_20": lambda: cand_trend_gated_mom_20(close),
        "smooth_mom_20": lambda: cand_smooth_mom_20(close),
        "zscore_20d": lambda: cand_zscore_20d(close),
        "breakout_20d": lambda: cand_breakout_20d(close),
        "updown_vol_asym_20": lambda: cand_updown_vol_asym_20(close),
        "roll_vol_ratio_20x60": lambda: cand_roll_vol_ratio_20x60(close),
        "cross_dispersion_60": lambda: cand_cross_dispersion_60(close),
        "xau_beta_cond_20": lambda: cand_xau_beta_cond_20(close),
        "us10y_mom_link_20": lambda: cand_us10y_mom_link_20(close, macro),
        "wti_link_20": lambda: cand_wti_link_20(close),
        "pos_return_ratio_20": lambda: cand_pos_return_ratio_20(close),
        "vol_gated_beta_60": lambda: cand_vol_gated_beta_60(close),
        # round-22 new
        "gap_ret_20": lambda: cand_gap_ret_20(close, open_),
        "close_loc_20": lambda: cand_close_loc_20(close, high, low),
        "sharpe_60d": lambda: cand_sharpe_60d(close),
        "dd_recovery_20": lambda: cand_dd_recovery_20(close),
        "us10y_rs_20": lambda: cand_us10y_rs_20(close, macro),
        "xau_corr_60": lambda: cand_xau_corr_60(close),
        "range_position_5": lambda: cand_range_position_5(close),
        "intraday_vol_share_20": lambda: cand_intraday_vol_share_20(close, high, low),
        "vol_beta_60": lambda: cand_vol_beta_60(close),
        "mom_consistency_60": lambda: cand_mom_consistency_60(close),
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
        print(f"{name:<24} IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} maxcorr={res_['max_abs_library_correlation']:.3f} cov_ge8={res_['coverage_dates_ge8']:.2f} turn={res_['turnover_10d_rank']:.2f} -> {'PASS' if res_['PASS'] else 'FAIL'}")
