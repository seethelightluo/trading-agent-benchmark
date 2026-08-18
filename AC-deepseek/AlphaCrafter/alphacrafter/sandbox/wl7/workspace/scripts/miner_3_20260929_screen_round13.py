"""
miner_3 cycle 2026-09-29: screen round 13.
Fixes pandas 3.0 stack(dropna=...) incompatibility that killed round 12,
re-runs round-12 candidates (13) that were never validated, adds new round-13
candidates (10), and re-validates the 8-factor library with a live-window drift check.

Library (8): rel_mom_20d_skip5, beta_ew_60d, downside_vol_ratio_20, max_ret_20d,
             eurusd_beta_cond_60x20, corr_ew_60, dxy_beta_cond_60x20, kurt_20d_skip5

Round-12 candidates (previously errored):
  us10y_beta_cond_60x20, cn10y_beta_cond_60x20, parkinson_ratio_20, gk_ratio_20,
  trend_r2_60, bollinger_pos_20, up_vol_20, downside_beta_ew_60, skew_20_skip5,
  xau_beta_riskoff_60x20, vol_rank_cs_20, hl_position_20, streak_5

New round-13 candidates:
  autocorr_5_60      - autocorrelation of 5d returns (trend persistence)
  drawdown_60        - distance from 60d high (mean reversion / drawdown)
  di_plus_minus_14   - directional movement index (ADX family, Wilder-lite)
  vix_regime_mom_20x60 - 20d momentum gated by VIX regime (calm markets)
  xau_beta_cond_60x20  - gold beta * gold 20d move (safe-haven rotation)
  wti_beta_cond_60x20  - oil beta * oil 20d move (growth/inflation rotation)
  crypto_beta_cond_60x20 - crypto beta * crypto 20d move (risk sentiment)
  overnight_gap_20   - mean open/prev-close gap over 20d
  range_breakout_20  - close vs 20d high (breakout)
  volume_trend_20    - 5d/20d average volume ratio (expansion)
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
    f = factor.stack().rename("f")          # pandas 3.0: no dropna arg
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
    lib["dxy_beta_cond_60x20"] = -fx_cond(macro["DXY"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-12 candidates (fixed) ----------------
def cand_yield_beta_cond(close, ref):
    ref20 = (ref / ref.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (-beta * ref20.reindex(s.index))
    return pd.DataFrame(cols, index=close.index)


def cand_parkinson_ratio_20(close, high, low):
    def f(s, h, l):
        rr = s.pct_change()
        cc = rr.rolling(20).std()
        hl = (np.log(h) - np.log(l))
        pk = np.sqrt((hl ** 2).rolling(20).mean() / (4 * np.log(2)))
        return pk / cc
    cols = {}
    for a in close.columns:
        s, h, l = close[a].dropna(), high[a].dropna(), low[a].dropna()
        idx = s.index.intersection(h.index).intersection(l.index)
        cols[a] = f(s.loc[idx], h.loc[idx], l.loc[idx])
    return pd.DataFrame(cols, index=close.index)


def cand_gk_ratio_20(close, open_, high, low):
    def f(s, o, h, l):
        rr = s.pct_change()
        cc = rr.rolling(20).std()
        lo = np.log(l / o)
        ho = np.log(h / o)
        co = np.log(s / o)
        gk = np.sqrt(0.5 * (ho ** 2 - 2 * lo * ho - (2 * np.log(2) - 1) * co ** 2)).rolling(20).mean()
        return gk / cc
    cols = {}
    for a in close.columns:
        s, o, h, l = (close[a].dropna(), open_[a].dropna(), high[a].dropna(), low[a].dropna())
        idx = s.index.intersection(o.index).intersection(h.index).intersection(l.index)
        cols[a] = f(s.loc[idx], o.loc[idx], h.loc[idx], l.loc[idx])
    return pd.DataFrame(cols, index=close.index)


def cand_trend_r2_60(close):
    def f(s):
        x = np.arange(60)
        def r2(y):
            if len(y) < 40 or np.std(y) == 0:
                return np.nan
            b = np.polyfit(x, y, 1)
            pred = np.polyval(b, x)
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return s.rolling(60).apply(r2, raw=True)
    return per_asset(f)(close)


def cand_bollinger_pos_20(close):
    def f(s):
        sma = s.rolling(20).mean()
        sd = s.rolling(20).std()
        return (s - sma) / (2 * sd)
    return per_asset(f)(close)


def cand_up_vol_20(close):
    def f(s):
        rr = s.pct_change()
        up = rr.where(rr > 0, np.nan)
        return up.rolling(20).std()
    return per_asset(f)(close)


def cand_downside_beta_ew_60(close):
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), ew_r.rename("m")], axis=1).dropna()
        z = z[z["m"] < 0]
        return z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    return per_asset(f)(close)


def cand_skew_20_skip5(close):
    def f(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).skew()
    return per_asset(f)(close)


def cand_xau_beta_riskoff_60x20(close, macro):
    vix = macro["VIX"].dropna()
    vix20 = (vix / vix.shift(20) - 1.0)
    g = close["XAU"].dropna()
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), g.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (beta * vix20.reindex(s.index))
    return pd.DataFrame(cols, index=close.index)


def cand_vol_rank_cs_20(close):
    v = close.pct_change().rolling(20).std()
    return v.rank(axis=1, pct=True)


def cand_hl_position_20(close, high, low):
    cols = {}
    for a in close.columns:
        s, h, l = close[a].dropna(), high[a].dropna(), low[a].dropna()
        idx = s.index.intersection(h.index).intersection(l.index)
        s2, h2, l2 = s.loc[idx], h.loc[idx], l.loc[idx]
        rng = h2.rolling(20).max() - l2.rolling(20).min()
        cols[a] = (s2 - l2.rolling(20).min()) / rng
    return pd.DataFrame(cols, index=close.index)


def cand_streak_5(close):
    def f(s):
        r = (s.pct_change() > 0).astype(float)
        out = []
        for i in range(59, len(r)):
            seg = r.iloc[i - 59:i + 1]
            runs = []
            cur = 0
            for v in seg:
                if v == 1:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)
            out.append(np.mean(runs) if runs else 0.0)
        return pd.Series(out, index=r.index[59:])
    return per_asset(f)(close)


# ---------------- new round-13 candidates ----------------
def cand_autocorr_5_60(close):
    """autocorrelation of 5d returns over a 60-observation window: trend persistence."""
    def f(s):
        rr = s.pct_change(5)
        def ac(y):
            if len(y) < 30 or np.std(y) == 0:
                return np.nan
            return np.corrcoef(y[:-1], y[1:])[0, 1]
        return rr.rolling(60).apply(ac, raw=True)
    return per_asset(f)(close)


def cand_drawdown_60(close):
    """current distance from 60d rolling high (negative = in drawdown)."""
    def f(s):
        return s / s.rolling(60).max() - 1.0
    return per_asset(f)(close)


def cand_di_plus_minus_14(close, high, low):
    """(DI+ - DI-) / (DI+ + DI-) with 14d smoothing: directional strength."""
    cols = {}
    for a in close.columns:
        s, h, l = close[a].dropna(), high[a].dropna(), low[a].dropna()
        idx = s.index.intersection(h.index).intersection(l.index)
        s2, h2, l2 = s.loc[idx], h.loc[idx], l.loc[idx]
        up = h2.diff()
        dn = -l2.diff()
        plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=s2.index)
        minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=s2.index)
        tr = pd.concat([(h2 - l2), (h2 - s2.shift()).abs(), (l2 - s2.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        pdi = 100 * plus_dm.rolling(14).mean() / atr
        mdi = 100 * minus_dm.rolling(14).mean() / atr
        denom = (pdi + mdi).replace(0, np.nan)
        cols[a] = (pdi - mdi) / denom
    return pd.DataFrame(cols, index=close.index)


def cand_vix_regime_mom_20x60(close, macro):
    """20d momentum (skip 5) active only in calm VIX regime (VIX below its 60d median)."""
    vix = macro["VIX"].dropna()
    calm = (vix < vix.rolling(60).median()).astype(float).reindex(close.index).ffill()
    mom = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    return mom.mul(calm, axis=0)


def cand_asset_beta_cond(close, ref):
    """beta to a reference asset * reference 20d move (rotation signal)."""
    ref20 = (ref / ref.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = beta * ref20.reindex(s.index)
    return pd.DataFrame(cols, index=close.index)


def cand_overnight_gap_20(close, open_):
    """mean open/prev-close gap over 20d (overnight vs intraday pattern)."""
    def f(s, o):
        gap = o / s.shift() - 1.0
        return gap.rolling(20).mean()
    cols = {}
    for a in close.columns:
        s, o = close[a].dropna(), open_[a].dropna()
        idx = s.index.intersection(o.index)
        cols[a] = f(s.loc[idx], o.loc[idx])
    return pd.DataFrame(cols, index=close.index)


def cand_range_breakout_20(close):
    """close vs 20d rolling high minus 1 (breakout proximity)."""
    def f(s):
        return s / s.rolling(20).max() - 1.0
    return per_asset(f)(close)


def cand_volume_trend_20(close, vol):
    """5d avg volume / 20d avg volume (volume expansion)."""
    cols = {}
    for a in close.columns:
        s, v = close[a].dropna(), vol[a].dropna()
        idx = s.index.intersection(v.index)
        v2 = v.loc[idx]
        cols[a] = v2.rolling(5).mean() / v2.rolling(20).mean()
    return pd.DataFrame(cols, index=close.index)


# ---------------- validation ----------------
def stacked_corr(cand, libsig):
    out = {}
    f = cand.stack().rename("f")
    for fid, ls in libsig.items():
        g = ls.stack().rename("g")
        j = pd.concat([f, g], axis=1).dropna()
        if len(j) < 100:
            out[fid] = float("nan")
            continue
        r = j["f"].corr(j["g"], method="spearman")
        out[fid] = float(r) if np.isfinite(r) else float("nan")
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
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.2f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    print(f"  -> {'PASS' if res['PASS'] else 'FAIL'} (gate:{gate} corr:{lowcorr})")
    print()
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets, warm-up through {WARM_END}, data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}")
    print(f"volume NaN frac: {vol.isna().mean().mean():.4f}; zero frac: {(vol == 0).mean().mean():.4f}")
    print()

    # ---- A) re-validate 8 effective library factors (warm-up) + live drift ----
    print("##### A) LIBRARY RE-VALIDATION #####")
    lib_results = {}
    for fid, sig in libsig.items():
        lib_results[fid] = validate(f"[LIB] {fid}", sig, close, libsig)
    print("##### A2) LIVE-WINDOW DRIFT (informational: 2026-07-16..end) #####")
    live_slice = close.loc[LIVE_START:]
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

    # ---- B) candidate screens ----
    print("##### B) CANDIDATE SCREENS #####")
    cands = {
        # round-12 (previously errored)
        "us10y_beta_cond_60x20": lambda: cand_yield_beta_cond(close, macro.get("US10Y")),
        "cn10y_beta_cond_60x20": lambda: cand_yield_beta_cond(close, macro.get("CN10Y")),
        "parkinson_ratio_20": lambda: cand_parkinson_ratio_20(close, high, low),
        "gk_ratio_20": lambda: cand_gk_ratio_20(close, open_, high, low),
        "trend_r2_60": lambda: cand_trend_r2_60(close),
        "bollinger_pos_20": lambda: cand_bollinger_pos_20(close),
        "up_vol_20": lambda: cand_up_vol_20(close),
        "downside_beta_ew_60": lambda: cand_downside_beta_ew_60(close),
        "skew_20_skip5": lambda: cand_skew_20_skip5(close),
        "xau_beta_riskoff_60x20": lambda: cand_xau_beta_riskoff_60x20(close, macro),
        "vol_rank_cs_20": lambda: cand_vol_rank_cs_20(close),
        "hl_position_20": lambda: cand_hl_position_20(close, high, low),
        "streak_5": lambda: cand_streak_5(close),
        # round-13 new
        "autocorr_5_60": lambda: cand_autocorr_5_60(close),
        "drawdown_60": lambda: cand_drawdown_60(close),
        "di_plus_minus_14": lambda: cand_di_plus_minus_14(close, high, low),
        "vix_regime_mom_20x60": lambda: cand_vix_regime_mom_20x60(close, macro),
        "xau_beta_cond_60x20": lambda: cand_asset_beta_cond(close, close["XAU"].dropna()),
        "wti_beta_cond_60x20": lambda: cand_asset_beta_cond(close, close["WTI"].dropna()),
        "crypto_beta_cond_60x20": lambda: cand_asset_beta_cond(close, close[["BTC", "ETH"]].mean(axis=1).dropna()),
        "overnight_gap_20": lambda: cand_overnight_gap_20(close, open_),
        "range_breakout_20": lambda: cand_range_breakout_20(close),
        "volume_trend_20": lambda: cand_volume_trend_20(close, vol),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===\n")

    print("##### SUMMARY #####")
    for name, r in results.items():
        print(f"{name:<26} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} cov_ge8={r['coverage_dates_ge8']:.2f} turn={r['turnover_10d_rank']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
