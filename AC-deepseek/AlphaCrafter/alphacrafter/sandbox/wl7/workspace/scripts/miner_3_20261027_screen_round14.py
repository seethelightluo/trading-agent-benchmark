"""
miner_3 cycle 2026-10-27: screen round 14.
Re-validates the 8-factor library (warm-up + live drift through 2026-10-26)
and screens fresh round-14 candidates. Uses the same admission gates:
|IC_h10| >= 0.007, |ICIR_h10| >= 0.084, max_abs_library_correlation < 0.5.

Round-14 candidates (fresh ideas, not previously screened):
  rsi_14_skip5        - classic RSI with 5d skip (trend/mean-reversion gauge)
  vol_expansion_10x60 - 10d realized vol / 60d realized vol (vol term structure)
  range_intraday_20   - mean (high-low)/close over 20d (intraday range proxy)
  reversal_5_skip2    - short-term reversal: -(5d ret skip 2) cross-sectional
  corr_chg_20x60      - 20d change of 60d EW correlation (comovement dynamics)
  trend_quality_20x60 - 20d mom (skip5) * 60d trend R2 (quality of trend)
  max_dd_20           - 20d max drawdown (risk / oversold)
  vol_price_corr_20   - rolling corr(volume, return) over 20d (volume confirmation)
  yield_mom_60        - 60d change of US10Y / CN10Y applied cross-sectionally
  vix_level_rank_20   - asset 20d vol rank gated by VIX level regime
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


# ---------------- round-14 candidates ----------------
def cand_rsi_14_skip5(close):
    def f(s):
        delta = s.diff()
        up = delta.clip(lower=0.0).rolling(14).mean()
        dn = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = up / dn.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.shift(5)
    return per_asset(f)(close)


def cand_vol_expansion_10x60(close):
    def f(s):
        rr = s.pct_change()
        v10 = rr.rolling(10).std()
        v60 = rr.rolling(60).std()
        return -(v10 / v60)  # low short vol vs long vol favored (contraction)
    return per_asset(f)(close)


def cand_range_intraday_20(close, high, low):
    cols = {}
    for a in close.columns:
        s, h, l = close[a].dropna(), high[a].dropna(), low[a].dropna()
        idx = s.index.intersection(h.index).intersection(l.index)
        rng = (h.loc[idx] - l.loc[idx]) / s.loc[idx]
        cols[a] = rng.rolling(20).mean()
    return pd.DataFrame(cols, index=close.index)


def cand_reversal_5_skip2(close):
    def f(s):
        return -(s.shift(2) / s.shift(7) - 1.0)  # short-term reversal
    return per_asset(f)(close)


def cand_corr_chg_20x60(close):
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()
    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        c = z["r"].rolling(60).corr(z["m"])
        return c - c.shift(20)
    return per_asset(f)(close)


def cand_trend_quality_20x60(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    def r2(s):
        x = np.arange(60)
        def _r2(y):
            if len(y) < 40 or np.std(y) == 0:
                return np.nan
            b = np.polyfit(x, y, 1)
            pred = np.polyval(b, x)
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return s.rolling(60).apply(_r2, raw=True)
    return m20 * per_asset(r2)(close)


def cand_max_dd_20(close):
    def f(s):
        roll_max = s.rolling(20, min_periods=10).max()
        return s / roll_max - 1.0  # negative in drawdown; favor recovery candidates
    return per_asset(f)(close)


def cand_vol_price_corr_20(close, vol):
    cols = {}
    for a in close.columns:
        s, v = close[a].dropna(), vol[a].dropna()
        idx = s.index.intersection(v.index)
        r = s.loc[idx].pct_change()
        vv = v.loc[idx]
        cols[a] = r.rolling(20).corr(vv)
    return pd.DataFrame(cols, index=close.index)


def cand_yield_mom_60(close):
    """cross-asset yield trend: 60d change of US10Y and CN10Y as conditional signals."""
    u10 = close["US10Y"].dropna()
    c10 = close["CN10Y"].dropna()
    du = (u10 / u10.shift(60) - 1.0)
    dc = (c10 / c10.shift(60) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        if a in ("US10Y", "CN10Y"):
            z = pd.concat([s.pct_change().rename("r"), s.pct_change().rename("x")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
            ref20 = (s / s.shift(60) - 1.0)
            cols[a] = (-beta * ref20.reindex(s.index))
            continue
        z = pd.concat([s.pct_change().rename("r"), du.reindex(s.index).rename("x")], axis=1).dropna()
        beta_u = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        z2 = pd.concat([s.pct_change().rename("r"), dc.reindex(s.index).rename("x")], axis=1).dropna()
        beta_c = z2["r"].rolling(60).cov(z2["x"]) / z2["x"].rolling(60).var()
        cols[a] = (-beta_u * du.reindex(s.index) - beta_c * dc.reindex(s.index))
    return pd.DataFrame(cols, index=close.index)


def cand_vix_level_rank_20(close, macro):
    vix = macro["VIX"].dropna()
    high_vix = (vix > vix.rolling(60).median()).astype(float).reindex(close.index).ffill()
    v = close.pct_change().rolling(20).std()
    rank = v.rank(axis=1, pct=True)
    return rank.mul(high_vix, axis=0)  # in high-vol regime, favor low-vol assets


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
    print()

    print("##### A) LIBRARY RE-VALIDATION (warm-up) #####")
    lib_results = {}
    for fid, sig in libsig.items():
        lib_results[fid] = validate(f"[LIB] {fid}", sig, close, libsig)

    print("##### A2) LIVE-WINDOW DRIFT (informational: 2026-07-16..end) #####")
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

    print("##### B) CANDIDATE SCREENS (round 14) #####")
    cands = {
        "rsi_14_skip5": lambda: cand_rsi_14_skip5(close),
        "vol_expansion_10x60": lambda: cand_vol_expansion_10x60(close),
        "range_intraday_20": lambda: cand_range_intraday_20(close, high, low),
        "reversal_5_skip2": lambda: cand_reversal_5_skip2(close),
        "corr_chg_20x60": lambda: cand_corr_chg_20x60(close),
        "trend_quality_20x60": lambda: cand_trend_quality_20x60(close),
        "max_dd_20": lambda: cand_max_dd_20(close),
        "vol_price_corr_20": lambda: cand_vol_price_corr_20(close, vol),
        "yield_mom_60": lambda: cand_yield_mom_60(close),
        "vix_level_rank_20": lambda: cand_vix_level_rank_20(close, macro),
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
        print(f"{name:<24} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} cov_ge8={r['coverage_dates_ge8']:.2f} turn={r['turnover_10d_rank']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
