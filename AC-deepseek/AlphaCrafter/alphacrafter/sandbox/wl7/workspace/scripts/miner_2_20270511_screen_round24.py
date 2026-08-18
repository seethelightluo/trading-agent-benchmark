"""
miner_2 cycle 2027-05-11: screen round 24 (data thru 2027-05-10).
Regime (per 04-27 ensemble notes + fresh data): VIX 9->17 spike onset (vol regime
LOW->HIGH transition), bond rout (US10Y/CN10Y breaking below MA20), FX trend
reversal (DXY 101->99.7, EURUSD rallying, USDJPY 162.8->161.0 rolling over),
mean |pairwise corr| rising 0.166->0.197, dispersion high (1.9%/day), XAU
resilient (only healthy defensive), frozen feeds 000688.SH/NDX/SOX.

Round-24 candidates (tailored to reversal/stress regime):
  usdjpy_beta_cond_60x20  - beta to USDJPY ret * USDJPY 20d ret (carry-unwind leg)
  usdcny_beta_cond_60x20  - beta to USDCNY ret * USDCNY 20d ret (China/CNY link)
  updown_vol_ratio_60     - upside/downside vol asymmetry over 60d
  ret_autocorr_5_20       - 20d return autocorrelation at lag 5 (trend consistency)
  dd_depth_60             - 60d max drawdown depth (risk aversion)
  xau_beta_60             - 60d beta to XAU (safe-haven beta)
  mom_60d_voladj_skip5    - 60d momentum / 60d vol (risk-adjusted intermediate mom)
  rsi_14                  - classic 14d RSI (oscillator mean-reversion)
  vol_skew_60             - percentile-based return asymmetry (P90-P50)/(P50-P10)
  wti_beta_60             - 60d beta to WTI (commodity beta, WTI in downtrend)

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


# ---------------- current library signals (8 effective) ----------------
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


# ---------------- round-24 candidates ----------------
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


def cand_updown_vol_ratio_60(close):
    r = close.pct_change()

    def ratio(w):
        w = np.asarray(w, dtype=float)
        w = w[~np.isnan(w)]
        if len(w) < 30:
            return np.nan
        up = w[w > 0] ** 2
        dn = w[w < 0] ** 2
        if dn.sum() <= 0:
            return np.nan
        return float(up.mean() / dn.mean())
    return r.rolling(60, min_periods=30).apply(ratio, raw=True)


def cand_ret_autocorr_5_20(close):
    r = close.pct_change()

    def ac5(w):
        w = np.asarray(w, dtype=float)
        w = w[~np.isnan(w)]
        if len(w) < 16:
            return np.nan
        x = w[5:]
        y = w[:-5]
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    return r.rolling(20, min_periods=12).apply(ac5, raw=True)


def cand_dd_depth_60(close):
    max60 = close.rolling(60, min_periods=30).max()
    dd = close / max60 - 1.0
    return dd.rolling(60, min_periods=30).min()


def cand_xau_beta_60(close):
    r = close.pct_change()
    xau_r = r["XAU"].dropna()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), xau_r.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["x"]) / z["x"].rolling(60, min_periods=30).var()
        out[a] = beta.reindex(close.index)
    return out


def cand_mom_60d_voladj_skip5(close):
    r = close.pct_change()
    m60 = close.shift(5) / close.shift(65) - 1.0
    v60 = r.rolling(60, min_periods=30).std()
    return m60 / (v60 + 1e-9)


def cand_rsi_14(close):
    r = close.pct_change()
    gain = r.clip(lower=0).rolling(14, min_periods=8).mean()
    loss = (-r.clip(upper=0)).rolling(14, min_periods=8).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def cand_vol_skew_60(close):
    r = close.pct_change()

    def skew_pct(w):
        w = np.asarray(w, dtype=float)
        w = w[~np.isnan(w)]
        if len(w) < 40:
            return np.nan
        p90, p50, p10 = np.percentile(w, [90, 50, 10])
        if (p50 - p10) <= 0:
            return np.nan
        return float((p90 - p50) / (p50 - p10))
    return r.rolling(60, min_periods=40).apply(skew_pct, raw=True)


def cand_wti_beta_60(close):
    r = close.pct_change()
    wti_r = r["WTI"].dropna()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), wti_r.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["x"]) / z["x"].rolling(60, min_periods=30).var()
        out[a] = beta.reindex(close.index)
    return out


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
    for s in ["US10Y", "CN10Y", "XAU", "BTC", "ETH", "WTI", "COPPER", "SPX", "NDX", "SOX", "000300.SH", "000688.SH", "N225", "HSI", "SX5E"]:
        if s in close.columns:
            print(f"{s} 20d={r[s].iloc[-20:].add(1).prod()-1:+.1%} 60d={r[s].iloc[-60:].add(1).prod()-1:+.1%}")
    mcorr = r.iloc[-60:].corr().abs().stack()
    mcorr = mcorr[mcorr < 0.999]
    print(f"mean |pairwise corr| last 60d: {mcorr.mean():.4f}")
    disp = r.sub(r.mean(axis=1), axis=0).abs().mean(axis=1)
    print(f"cross-sectional daily dispersion last 20d mean: {disp.iloc[-20:].mean()*100:.2f}%")
    print()

    print("##### A1) LIBRARY WARM-UP REVALIDATION (thru 2026-07-15) #####")
    for fid, sig in libsig.items():
        validate(fid, sig, close, libsig, report_live=True)

    print("##### B) ROUND-24 CANDIDATES #####")
    cands = {
        "usdjpy_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "USDJPY"),
        "usdcny_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "USDCNY"),
        "updown_vol_ratio_60": lambda: cand_updown_vol_ratio_60(close),
        "ret_autocorr_5_20": lambda: cand_ret_autocorr_5_20(close),
        "dd_depth_60": lambda: cand_dd_depth_60(close),
        "xau_beta_60": lambda: cand_xau_beta_60(close),
        "mom_60d_voladj_skip5": lambda: cand_mom_60d_voladj_skip5(close),
        "rsi_14": lambda: cand_rsi_14(close),
        "vol_skew_60": lambda: cand_vol_skew_60(close),
        "wti_beta_60": lambda: cand_wti_beta_60(close),
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
