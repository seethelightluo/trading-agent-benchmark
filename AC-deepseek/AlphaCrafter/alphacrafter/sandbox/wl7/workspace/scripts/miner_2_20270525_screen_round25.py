"""
miner_2 cycle 2027-05-25: screen round 25 (data thru 2027-05-24).
Regime (fresh probe): VIX 29.37 (+71.1%/20d, +226.3%/60d) - HIGH and STILL
RISING vol regime (9.0 pin -> escalation, 2026 9->47 precedent). BTC -20.4%/20d
c/MA20 0.822 deep drawdown, WTI -3.5%, 000300.SH -6.5%; defensive bid: CN10Y
+11.7% (c/MA20 1.061), US10Y +6.6%, XAU +4.7% resilient; risk assets HSI +8.6%,
COPPER +7.4%, N225 +4.9%, SX5E +3.1%. Mean |pairwise corr| 20d 0.199 (elevated),
dispersion 1.27%/day. Frozen feeds 000688.SH/NDX/SOX.

Round-25 candidates (stress/trend-hygiene/risk-off family, distinct from round-24):
  dist_252d_high         - close / rolling_max(close,252): 1y-high proximity (George-Hwang)
  skew_20d_skip5         - 20d return skewness lag5 (tail asymmetry, kurt analog)
  vol_ratio_5_20         - 5d vol / 20d vol (vol acceleration)
  close_pos_20d          - (close-min20)/(max20-min20): 20d range position
  recovery_ratio_60      - close / rolling_min(close,60): drawdown recovery
  ma_cross_10_40         - SMA10/SMA40-1: trend crossover
  coskew_ew_60           - -systematic coskewness vs equal-weight index (Harvey-Siddique)
  downside_vol_ratio_60  - 60d downside-vol ratio variant (longer window)
  xau_beta_cond_60x20    - beta to XAU * XAU 20d trend (defensive beta conditioned)
  copper_beta_cond_60x20 - beta to COPPER * COPPER 20d trend (growth beta conditioned)
  vix_cond_mom_20        - rel_mom_20d_skip5 sign-flipped when VIX 20d rising (regime-cond mom)
  rev_20d_voladj         - -(20d return)/20d vol (vol-adjusted reversal)

Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084, max_abs_library_corr < 0.5.
Primary validation window: warm-up 2020-01-01..2026-07-15; live drift informational.
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


# ---------------- current library signals (8) ----------------
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
    lib["dxy_beta_cond_60x20"] = fx_cond(macro["DXY"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-25 candidates ----------------
def cand_dist_252d_high(close):
    return close / close.rolling(252, min_periods=120).max()


def cand_skew_20d_skip5(close):
    r = close.pct_change().shift(5)
    return r.rolling(20, min_periods=12).skew()


def cand_vol_ratio_5_20(close):
    r = close.pct_change()
    v5 = r.rolling(5, min_periods=4).std()
    v20 = r.rolling(20, min_periods=12).std()
    return v5 / (v20 + 1e-12)


def cand_close_pos_20d(close):
    hi = close.rolling(20, min_periods=10).max()
    lo = close.rolling(20, min_periods=10).min()
    return (close - lo) / (hi - lo + 1e-12)


def cand_recovery_ratio_60(close):
    mn = close.rolling(60, min_periods=30).min()
    return close / mn - 1.0


def cand_ma_cross_10_40(close):
    s10 = close.rolling(10, min_periods=6).mean()
    s40 = close.rolling(40, min_periods=20).mean()
    return s10 / s40 - 1.0


def cand_coskew_ew_60(close):
    r = close.pct_change()
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ew_r.reindex(s.index).rename("m")], axis=1).dropna()
        r_ = z["r"]
        m_ = z["m"]

        def csk(w):
            w = np.asarray(w, dtype=float)
            if len(w) < 30:
                return np.nan
            rw, mw = w[:, 0], w[:, 1]
            if np.std(rw) == 0 or np.std(mw) == 0:
                return np.nan
            return float(np.mean((rw - rw.mean()) * (mw - mw.mean()) ** 2) /
                         (np.std(rw) * np.std(mw) ** 2 + 1e-12))
        z["csk"] = z[["r", "m"]].rolling(60, min_periods=30).apply(csk, raw=True)
        out[a] = -z["csk"].reindex(close.index)  # negative coskew premium
    return out


def cand_downside_vol_ratio_60(close):
    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(60, min_periods=30).mean())
        tot = rr.rolling(60, min_periods=30).std()
        return -(ds / tot)
    return per_asset(dsvr)(close)


def cand_ref_beta_cond(close, ref_name, ref_series, window=60, cond=20):
    ref = ref_series.dropna()
    ref_ret = ref.pct_change()
    ref_cond = (ref / ref.shift(cond) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref_ret.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(window).cov(z["x"]) / z["x"].rolling(window).var()
        out[a] = (beta * ref_cond.reindex(s.index)).reindex(close.index)
    return out


def cand_vix_cond_mom_20(close, macro):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    rel = m20.sub(m20.median(axis=1), axis=0)
    vix = macro["VIX"].dropna()
    vix_up = (vix / vix.shift(20) - 1.0) > 0
    flip = pd.Series(np.where(vix_up.reindex(rel.index).fillna(False), -1.0, 1.0), index=rel.index)
    return rel.mul(flip, axis=0)


def cand_rev_20d_voladj(close):
    r = close.pct_change()
    m20 = close / close.shift(20) - 1.0
    v20 = r.rolling(20, min_periods=12).std()
    return -(m20 / (v20 + 1e-12))


def validate(name, factor, close, libsig, window_end=WARM_END, report_live=True):
    res = {"n_dates": int(factor.loc[:window_end].shape[0])}
    ic = rank_ic_series(factor.loc[:window_end], fwd_returns(close, 10))
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
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}")
    if report_live:
        sub = factor.loc[LIVE_START:]
        if sub.notna().sum().sum() >= 200:
            ic_l = rank_ic_series(sub, fwd_returns(close, 10))
            if len(ic_l) >= 5:
                d = 1.0 if ic_l.mean() >= 0 else -1.0
                icir_l = d * ic_l.mean() / ic_l.std() if ic_l.std() > 0 else float("nan")
                print(f"  LIVE h10 IC={d*ic_l.mean():+.4f} ICIR={icir_l:+.4f} hit={(d*ic_l>0).mean():.3f} n={len(ic_l)}")
            else:
                print(f"  LIVE h10: n={len(ic_l)} too small")
        else:
            print("  LIVE: too few obs")
    print()
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}\n")

    r = close.pct_change()
    print("regime sanity: ", end="")
    for s in ["VIX", "DXY", "USDJPY", "EURUSD", "USDCNY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%} ", end="")
    print()
    mcorr = r.iloc[-20:].corr().abs().stack()
    mcorr = mcorr[mcorr < 0.999]
    print(f"mean |pairwise corr| last 20d: {mcorr.mean():.4f}")
    disp = r.sub(r.mean(axis=1), axis=0).abs().mean(axis=1)
    print(f"cross-sectional daily dispersion last 20d mean: {disp.iloc[-20:].mean()*100:.2f}%")
    print()

    print("##### A) LIBRARY WARM-UP REVALIDATION (thru 2026-07-15) #####")
    for fid, sig in libsig.items():
        validate(fid, sig, close, libsig, report_live=True)

    print("##### B) ROUND-25 CANDIDATES #####")
    xau = close["XAU"].dropna()
    copper = close["COPPER"].dropna()
    cands = {
        "dist_252d_high": lambda: cand_dist_252d_high(close),
        "skew_20d_skip5": lambda: cand_skew_20d_skip5(close),
        "vol_ratio_5_20": lambda: cand_vol_ratio_5_20(close),
        "close_pos_20d": lambda: cand_close_pos_20d(close),
        "recovery_ratio_60": lambda: cand_recovery_ratio_60(close),
        "ma_cross_10_40": lambda: cand_ma_cross_10_40(close),
        "coskew_ew_60": lambda: cand_coskew_ew_60(close),
        "downside_vol_ratio_60": lambda: cand_downside_vol_ratio_60(close),
        "xau_beta_cond_60x20": lambda: cand_ref_beta_cond(close, "XAU", xau),
        "copper_beta_cond_60x20": lambda: cand_ref_beta_cond(close, "COPPER", copper),
        "vix_cond_mom_20": lambda: cand_vix_cond_mom_20(close, macro),
        "rev_20d_voladj": lambda: cand_rev_20d_voladj(close),
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
