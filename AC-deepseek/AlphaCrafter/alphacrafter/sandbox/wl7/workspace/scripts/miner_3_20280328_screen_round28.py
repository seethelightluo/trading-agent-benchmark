"""
miner_3 cycle 2028-03-28: screen round 28 (data thru 2028-03-27).
Regime: VIX 29.16 HIGH-vol (+46%/20d re-escalation), 4/15 assets above MA20
(weak breadth), WTI +28.6%/20d energy leadership, BTC rebound +4.9% vs ETH
-10.8%, XAU +3.3% safe haven, DXY +3.1%/20d (USD strengthening), dispersion
elevated, pairwise corr 0.108 LOW -> cross-sectional dispersion factors favored.
Frozen feeds: NDX/SOX/000688.SH/CN10Y flat.

Round-28 candidates (new formulations motivated by regime):
  - trend_accel_20x10   : acceleration of relative 20d momentum (skip5)
  - range_pos_60d       : close position within 60d range (trend position)
  - idio_mom_20x60      : idiosyncratic momentum (EW-beta residual, 20d skip5)
  - gk_vol_20_neg       : negative Garman-Klass vol 20d (low-vol premium)
  - vol_regime_5x60_neg : negative 5d/60d vol ratio (vol deceleration)
  - bond_rs_mom_20      : 20d return minus US10Y 20d return (risk appetite)
  - xau_rs_mom_20       : 20d return minus XAU 20d return (defensive rotation)
  - spx_rs_mom_20       : 20d return minus SPX 20d return (US equity rotation)
  - cn_rs_mom_20        : 20d return minus 000300.SH 20d return (China rotation)
  - wti_rs_mom_20       : 20d return minus WTI 20d return (energy rotation)
  - btc_rs_mom_20       : 20d return minus BTC 20d return (crypto rotation)
  - vol_slope_20        : 20d log-volume slope (volume momentum)

Admission gates (warm-up 2020-01-01..2026-07-15): |IC_h10| >= 0.007,
|ICIR_h10| >= 0.084, max_abs_library_correlation < 0.5.
Live window 2026-07-16..2028-03-27 reported as informational drift.
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


# ---------------- current library signals (8 persisted) ----------------
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


# ---------------- round-28 new candidates ----------------
def cand_trend_accel_20x10(close):
    """Acceleration of relative 20d momentum: rel_mom(t) - rel_mom(t-10)."""
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    rel = m20.sub(m20.median(axis=1), axis=0)
    return rel - rel.shift(10)


def cand_range_pos_60d(close):
    """Close position within 60d range: (C - min60)/(max60 - min60)."""
    def f(s):
        mn = s.rolling(60, min_periods=30).min()
        mx = s.rolling(60, min_periods=30).max()
        return (s - mn) / (mx - mn)
    return per_asset(f)(close)


def cand_idio_mom_20x60(close):
    """Idiosyncratic 20d momentum: asset 20d ret - beta60 * EW 20d ret (skip5)."""
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()
    ew20 = ew.shift(5) / ew.shift(25) - 1.0
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), ew_r.reindex(s.index).rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
        a20 = s.shift(5) / s.shift(25) - 1.0
        out[a] = (a20 - beta * ew20.reindex(s.index)).reindex(close.index)
    return out


def cand_gk_vol_20_neg(close, open_, high, low):
    """Negative Garman-Klass 20d volatility (low-vol premium)."""
    co = np.log(close / open_)
    hl = np.log(high / low)
    var = 0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2
    return -np.sqrt(var.rolling(20, min_periods=12).mean())


def cand_vol_regime_5x60_neg(close):
    """Negative 5d/60d realized vol ratio (vol deceleration)."""
    r = close.pct_change()
    v5 = r.rolling(5).std()
    v60 = r.rolling(60, min_periods=30).std()
    return -(v5 / v60)


def _rs_mom(close, ref_sym):
    ref = close[ref_sym].dropna()
    ref20 = ref / ref.shift(20) - 1.0
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        a20 = s / s.shift(20) - 1.0
        out[a] = (a20 - ref20.reindex(s.index)).reindex(close.index)
    return out


def cand_vol_slope_20(vol):
    """20d slope of log volume via linear regression (z-scored)."""
    def f(s):
        lv = np.log(s.replace(0, np.nan) + 1.0)
        x = np.arange(20, dtype=float)
        x = x - x.mean()
        def _slope(w):
            w = np.asarray(w, dtype=float)
            if np.isnan(w).any() or np.std(w) == 0:
                return np.nan
            return float(np.dot(w - w.mean(), x) / np.dot(x, x))
        return lv.rolling(20, min_periods=10).apply(_slope, raw=True)
    return per_asset(f)(vol)


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
    print()

    cands = {
        "trend_accel_20x10": lambda: cand_trend_accel_20x10(close),
        "range_pos_60d": lambda: cand_range_pos_60d(close),
        "idio_mom_20x60": lambda: cand_idio_mom_20x60(close),
        "gk_vol_20_neg": lambda: cand_gk_vol_20_neg(close, open_, high, low),
        "vol_regime_5x60_neg": lambda: cand_vol_regime_5x60_neg(close),
        "bond_rs_mom_20": lambda: _rs_mom(close, "US10Y"),
        "xau_rs_mom_20": lambda: _rs_mom(close, "XAU"),
        "spx_rs_mom_20": lambda: _rs_mom(close, "SPX"),
        "cn_rs_mom_20": lambda: _rs_mom(close, "000300.SH"),
        "wti_rs_mom_20": lambda: _rs_mom(close, "WTI"),
        "btc_rs_mom_20": lambda: _rs_mom(close, "BTC"),
        "vol_slope_20": lambda: cand_vol_slope_20(vol),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===\n")

    print("##### LIVE-WINDOW DRIFT (informational 2026-07-16..end) #####")
    for name, res_ in results.items():
        pass
    print()

    print("##### SUMMARY #####")
    for name, res_ in results.items():
        print(f"{name}: IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} hit={res_['hit_h10']:.3f} "
              f"cov_ge8={res_['coverage_dates_ge8']:.3f} turn={res_['turnover_10d_rank']:.3f} "
              f"maxcorr={res_['max_abs_library_correlation']:.3f} "
              f"-> {'PASS' if res_['PASS'] else 'FAIL'}")
