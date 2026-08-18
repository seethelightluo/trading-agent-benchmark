"""
miner_3 cycle 2027-07-20: screen round 26 (data thru 2027-07-19).
1) Re-validates the 7-factor EFFECTIVE library (warm-up admission gate +
   live-window drift) to refresh stale last_validated timestamps.
2) Screens round-26 candidates focused on the LIVE regime:
   - VIX 18.2 normalizing DOWN (spring escalation resolved), risk MEDIUM
   - DXY 103.4 +1.9%/20d: strong-USD regime LIVE again -> FX-conditioned bets
     (USDJPY, USDCNY) complement dxy_beta_cond_60x20
   - Commodity complex (XAU/COPPER/WTI) beta-conditioning (safe-haven / global
     growth rotation)
   - Bond-equity rotation (US10Y beta conditioned on yield momentum)
   - Squeeze/breakout timing (vol contraction, proximity to range high)
   - Idiosyncratic move / cross-sectional dispersion capture
   Frozen feeds persist: 000688.SH/NDX/SOX flat (rank-neutral dead weight).

Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084,
max_abs_library_correlation < 0.5. Warm-up 2020-01-01..2026-07-15 is the
admission window; live 2026-07-16..end is informational drift.
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


# ---------------- round-26 new candidates ----------------
def cond_beta_factor(close, ref_series, ref_mom_window=20, beta_window=60):
    """Generic: asset beta to ref * ref 20d momentum (regime conditioning)."""
    ref = ref_series.dropna()
    ref20 = (ref / ref.shift(ref_mom_window) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(beta_window).cov(z["x"]) / z["x"].rolling(beta_window).var()
        out[a] = (beta * ref20.reindex(s.index)).reindex(close.index)
    return out


def cand_trend_mom_20x60(close):
    """trend-confirmed momentum: 20d mom(skip5) * (close/MA60 - 1)."""
    mom20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    ma60 = close.rolling(60, min_periods=30).mean()
    trend = close / ma60 - 1.0
    out = mom20 * trend
    return out.sub(out.median(axis=1), axis=0)


def cand_squeeze_5x20(close):
    """volatility squeeze: 5d realized vol / 20d realized vol (low = coiling)."""
    r = close.pct_change()
    v5 = r.rolling(5, min_periods=4).std()
    v20 = r.rolling(20, min_periods=12).std()
    return v5 / (v20 + 1e-9)


def cand_breakout_prox_20(close):
    """proximity to 20d high: close / rolling_max(close,20) (high = breakout)."""
    hi20 = close.rolling(20, min_periods=10).max()
    return close / hi20 - 1.0


def cand_idio_move_20(close):
    """idiosyncratic move: |ret - cross-sectional mean| averaged 20d."""
    r = close.pct_change()
    dev = (r - r.mean(axis=1)).abs()
    return dev.rolling(20, min_periods=10).mean()


def cand_vol_ratio_5x60(close):
    """vol regime: 5d vol / 60d vol (short-vol regime tilt)."""
    r = close.pct_change()
    v5 = r.rolling(5, min_periods=4).std()
    v60 = r.rolling(60, min_periods=30).std()
    return v5 / (v60 + 1e-9)


def cand_bond_stock_rot_60(close, macro):
    """bond-equity rotation: 60d corr with US10Y returns, conditioned on
    US10Y 20d momentum (bond selloff vs rally regime)."""
    u10 = close["US10Y"].dropna()
    u10_r = u10.pct_change()
    u10_20 = (u10 / u10.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), u10_r.reindex(s.index).rename("x")], axis=1).dropna()
        c = z["r"].rolling(60, min_periods=30).corr(z["x"])
        out[a] = (c * u10_20.reindex(s.index)).reindex(close.index)
    return out


def cand_xau_lead_20(close):
    """gold safe-haven spillover: 20d corr with XAU returns * XAU 20d mom."""
    xau = close["XAU"].dropna()
    xau_r = xau.pct_change()
    xau20 = (xau / xau.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), xau_r.reindex(s.index).rename("x")], axis=1).dropna()
        c = z["r"].rolling(60, min_periods=30).corr(z["x"])
        out[a] = (c * xau20.reindex(s.index)).reindex(close.index)
    return out


def cand_crypto_spill_20(close):
    """crypto spillover: 20d corr with BTC returns * BTC 20d mom."""
    btc = close["BTC"].dropna()
    btc_r = btc.pct_change()
    btc20 = (btc / btc.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), btc_r.reindex(s.index).rename("x")], axis=1).dropna()
        c = z["r"].rolling(60, min_periods=30).corr(z["x"])
        out[a] = (c * btc20.reindex(s.index)).reindex(close.index)
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

    print("##### B) ROUND-26 CANDIDATE SCREENS #####")
    cands = {
        "usdjpy_beta_cond_60x20": lambda: cond_beta_factor(close, macro["USDJPY"].dropna()),
        "usdcny_beta_cond_60x20": lambda: cond_beta_factor(close, macro["USDCNY"].dropna()),
        "trend_mom_20x60": lambda: cand_trend_mom_20x60(close),
        "squeeze_5x20": lambda: cand_squeeze_5x20(close),
        "breakout_prox_20": lambda: cand_breakout_prox_20(close),
        "idio_move_20": lambda: cand_idio_move_20(close),
        "vol_ratio_5x60": lambda: cand_vol_ratio_5x60(close),
        "bond_stock_rot_60": lambda: cand_bond_stock_rot_60(close, macro),
        "xau_lead_20": lambda: cand_xau_lead_20(close),
        "crypto_spill_20": lambda: cand_crypto_spill_20(close),
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
