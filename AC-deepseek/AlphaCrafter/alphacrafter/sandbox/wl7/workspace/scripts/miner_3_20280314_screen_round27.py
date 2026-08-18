"""
miner_3 cycle 2028-03-14: screen round 27 (data thru 2028-03-13).
1) Re-validates the 7-factor EFFECTIVE library (warm-up admission gate +
   live-window drift) and refreshes last_validated timestamps.
2) Screens round-27 candidates focused on the LIVE regime:
   - VIX ~21.9 re-escalating (cooled 27->20 then +18%/5d up again), vol MEDIUM-HIGH
   - XAU-led safe-haven leadership (XAU +6% last block, defensive floor working)
   - Commodity rotation (WTI +11% rebound, COPPER positive) - commodity momentum conditioning
   - SPX pullback (-1.2%/block) - US equity leadership conditioning
   - US10Y selloff (-1.4%) - bond regime conditioning
   - China recovery (000300.SH +4.2%, HSI +2.0%) - CN10Y bond flow conditioning
   - Risk-premium ideas: drawdown depth, return skewness (crash-risk), lag-1
     autocorrelation (persistence/mean-reversion), up-day breadth, RSI extremes
   Frozen feeds persist: NDX/SOX/000688/CN10Y flat (rank-neutral dead weight).

Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084,
max_abs_library_correlation < 0.5. Warm-up 2020-01-01..2026-07-15 is the
admission window; live 2026-07-16..2028-03-13 is informational drift.
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
    lib["dxy_beta_cond_60x20"] = fx_cond(macro["DXY"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-27 new candidates ----------------
def cond_beta_factor(close, ref_series, ref_mom_window=20, beta_window=60):
    """Generic: asset beta to ref * ref momentum (regime conditioning)."""
    ref = ref_series.dropna()
    refm = (ref / ref.shift(ref_mom_window) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(beta_window).cov(z["x"]) / z["x"].rolling(beta_window).var()
        out[a] = (beta * refm.reindex(s.index)).reindex(close.index)
    return out


def cand_drawdown_depth_60(close):
    """60d max drawdown from running peak, sign-flipped (higher = shallower drawdown)."""
    def f(s):
        peak = s.rolling(60, min_periods=30).max()
        return -(s / peak - 1.0)
    return per_asset(f)(close)


def cand_skew_20d_skip5(close):
    """20d return skewness with 5d skip (crash-risk asymmetry)."""
    def f(s):
        return s.pct_change().shift(5).rolling(20, min_periods=12).skew()
    return per_asset(f)(close)


def cand_autocorr_1_20(close):
    """lag-1 autocorrelation of daily returns over 20d (persistence vs mean-reversion)."""
    def f(s):
        rr = s.pct_change()
        a = rr.rolling(20, min_periods=10).apply(
            lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 3 else np.nan, raw=False)
        return a
    return per_asset(f)(close)


def cand_upday_ratio_20(close):
    """fraction of up days over 20d (breadth/tone)."""
    r = close.pct_change()
    return (r > 0).rolling(20, min_periods=10).mean()


def cand_rsi14(close):
    """RSI(14): classic mean-reversion at extremes."""
    def f(s):
        d = s.diff()
        up = d.clip(lower=0.0).rolling(14).mean()
        dn = (-d.clip(upper=0.0)).rolling(14).mean()
        rs = up / (dn + 1e-12)
        return 100.0 - 100.0 / (1.0 + rs)
    return per_asset(f)(close)


def cand_xau_ratio_beta_60x20(close):
    """safe-haven rotation: beta to XAU/COPPER ratio * ratio 20d momentum."""
    xau = close["XAU"].dropna()
    cu = close["COPPER"].dropna()
    ratio = (xau / cu).reindex(close.index)
    ratio = ratio.dropna()
    return cond_beta_factor(close, ratio, ref_mom_window=20, beta_window=60)


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
    libres = {}
    for fid, sig in libsig.items():
        libres[fid] = validate(fid, sig, close, libsig)

    print("##### A2) LIBRARY LIVE-WINDOW DRIFT (informational: 2026-07-16..end) #####")
    live_ic = {}
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
        live_ic[fid] = (d * ic.mean(), icir, (d * ic > 0).mean(), len(ic))
        print(f"  {fid}: live h10 IC={d*ic.mean():+.4f} ICIR={icir:+.4f} hit={(d*ic>0).mean():.3f} n={len(ic)}")
    print()

    print("##### B) ROUND-27 CANDIDATE SCREENS #####")
    cands = {
        "xau_beta_cond_60x20": lambda: cond_beta_factor(close, close["XAU"].dropna()),
        "wti_beta_cond_60x20": lambda: cond_beta_factor(close, close["WTI"].dropna()),
        "spx_beta_cond_60x20": lambda: cond_beta_factor(close, close["SPX"].dropna()),
        "cny10y_beta_cond_60x20": lambda: cond_beta_factor(close, close["CN10Y"].dropna()),
        "xau_ratio_beta_60x20": lambda: cand_xau_ratio_beta_60x20(close),
        "drawdown_depth_60": lambda: cand_drawdown_depth_60(close),
        "skew_20d_skip5": lambda: cand_skew_20d_skip5(close),
        "autocorr_1_20": lambda: cand_autocorr_1_20(close),
        "upday_ratio_20": lambda: cand_upday_ratio_20(close),
        "rsi14": lambda: cand_rsi14(close),
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
        print(f"{name}: IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} hit={res_['hit_h10']:.3f} "
              f"cov_ge8={res_['coverage_dates_ge8']:.3f} maxcorr={res_['max_abs_library_correlation']:.3f} "
              f"-> {'PASS' if res_['PASS'] else 'FAIL'}")
