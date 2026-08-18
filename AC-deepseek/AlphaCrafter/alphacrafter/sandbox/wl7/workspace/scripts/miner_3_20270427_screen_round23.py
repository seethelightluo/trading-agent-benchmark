"""
miner_3 cycle 2027-04-27: screen round 23 (data thru 2027-04-26).
Re-validates the 7-factor library and refreshes round-22 candidates with
fresh data; adds round-23 candidates designed for the current regime:
VIX 17.16 (+90.7%/20d - off the 9.0 floor, stress episode materializing),
BTC -16.5%/20d crash, SX5E -13.8%/20d weakest, US10Y +7.0% bond rally
(price terms), CN10Y -11.7% selloff, XAU +5.4% up, DXY -1.5% weakening,
mean |pairwise corr| 0.114 (low), dispersion 1.19%/day (moderate).

Admission gates (same as prior cycles): |IC_h10| >= 0.007, |ICIR_h10| >= 0.084,
max_abs_library_correlation < 0.5. Validation window warm-up through 2026-07-15
plus live drift 2026-07-16..end (informational).

Round-23 new candidates (stress/rotation regime):
  us10y_beta_cond_60x20 - beta to US10Y price * US10Y 20d ret (bond-rally leg)
  vix_beta_cond_60x20   - beta to VIX * VIX 20d change (VIX-spike hedgers)
  dxy_beta_cond_60x20   - beta to DXY * DXY 20d ret (USD-weakening leg)
  xau_rs_20             - asset 20d ret minus XAU 20d ret (safe-haven RS)
  tail_risk_5d          - mean of 5 worst daily rets over 60d (tail-aversion)
  skew_20d_skip5        - 20d skewness skip5 (lottery-aversion complement)
  vol_ratio_10x60       - 10d vol / 60d vol (short-term vol spike)
  bond_affinity_60      - 60d corr to US10Y price (bond-like assets)
  usd_affinity_60       - -60d corr to DXY (anti-USD assets)
  crash_gap_20          - 20d sum of negative gap returns (gap-down stress)
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


# ---------------- round-22 candidates (refresh) ----------------
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


# ---------------- round-23 new candidates ----------------
def cand_macro_beta_cond(close, macro, name, window=60, cond=20):
    ref = macro[name].dropna()
    ref_ret = ref.pct_change()
    ref_cond = (ref / ref.shift(cond) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref_ret.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(window).cov(z["x"]) / z["x"].rolling(window).var()
        out[a] = (beta * ref_cond.reindex(s.index)).reindex(close.index)
    return out


def cand_xau_rs_20(close):
    xau = close["XAU"].dropna()
    xau20 = (xau / xau.shift(20) - 1.0).reindex(close.index).ffill()
    m20 = close / close.shift(20) - 1.0
    return m20 - xau20


def cand_tail_risk_5d(close):
    r = close.pct_change()

    def worst5(w):
        w = np.asarray(w, dtype=float)
        w = w[~np.isnan(w)]
        if len(w) < 30:
            return np.nan
        return float(np.sort(w)[:5].mean())
    return r.rolling(60, min_periods=30).apply(worst5, raw=True)


def cand_skew_20d_skip5(close):
    def sk(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).skew()
    return per_asset(sk)(close)


def cand_vol_ratio_10x60(close):
    r = close.pct_change()
    v10 = r.rolling(10, min_periods=6).std()
    v60 = r.rolling(60, min_periods=30).std()
    return v10 / (v60 + 1e-9)


def cand_bond_affinity_60(close, macro):
    us10y = macro["US10Y"].dropna()
    us10y_r = us10y.pct_change()
    r = close.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = r[a].rolling(60, min_periods=30).corr(us10y_r.reindex(r.index))
    return out


def cand_usd_affinity_60(close, macro):
    dxy = macro["DXY"].dropna()
    dxy_r = dxy.pct_change()
    r = close.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = -r[a].rolling(60, min_periods=30).corr(dxy_r.reindex(r.index))
    return out


def cand_crash_gap_20(close, open_):
    gap = open_ / close.shift(1) - 1.0
    neg = gap.where(gap < 0, 0.0)
    return neg.rolling(20, min_periods=12).sum()


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

    print("##### B) CANDIDATE SCREENS (round 22 refresh + round 23 new) #####")
    cands = {
        # round-22 refresh
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
        # round-23 new
        "us10y_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "US10Y"),
        "vix_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "VIX"),
        "dxy_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "DXY"),
        "xau_rs_20": lambda: cand_xau_rs_20(close),
        "tail_risk_5d": lambda: cand_tail_risk_5d(close),
        "skew_20d_skip5": lambda: cand_skew_20d_skip5(close),
        "vol_ratio_10x60": lambda: cand_vol_ratio_10x60(close),
        "bond_affinity_60": lambda: cand_bond_affinity_60(close, macro),
        "usd_affinity_60": lambda: cand_usd_affinity_60(close, macro),
        "crash_gap_20": lambda: cand_crash_gap_20(close, open_),
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
        print(f"{name:<26} IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} maxcorr={res_['max_abs_library_correlation']:.3f} cov_ge8={res_['coverage_dates_ge8']:.2f} turn={res_['turnover_10d_rank']:.2f} -> {'PASS' if res_['PASS'] else 'FAIL'}")
