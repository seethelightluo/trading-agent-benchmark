"""
miner_3 cycle 2027-05-11: screen round 24 (data thru 2027-05-10).
Re-validates the 7-factor library and screens round-24 candidates for the
current regime:
  VIX 21.03 (spiked 9.0 -> ~24.8 mid-May block, now -15.1%/5d unwind; still
  elevated), SPX -7.9%/60d but +1.2%/5d bounce, BTC +16.5%/5d +49%/60d
  leadership, SX5E +20.6%/60d recovery, N225 +11.5%/60d, XAU -6.2%/20d
  (safe-haven fade), COPPER -5.9%/20d, WTI +7.4%/20d, US10Y -2.1%/20d
  (bond leg mixed), CN10Y -7%/60d bond rout, DXY flat 99.8, EURUSD -1.3%/20d,
  mean |corr| 20d 0.178 (rising), dispersion 1.56%/day (moderate).
  Frozen feeds: 000688.SH/SOX/NDX (0% returns, rank-neutral dead weight).

Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084,
max_abs_library_correlation < 0.5. Validation window warm-up thru 2026-07-15;
live drift 2026-07-16..end informational.

Round-24 new candidates (post-stress rotation / crypto-leadership / vol regime):
  rsi_14               - classic 14d RSI (mean-reversion)
  trend_r2_20          - R2 of 20d log-price vs time (trend consistency)
  dd_depth_20          - -(close/rolling_max(close,20)-1): short DD depth
  rel_mom_60d_skip5    - 60d cross-sectional relative momentum, skip5
  btc_rs_20            - asset 20d ret minus BTC 20d ret (crypto leadership)
  btc_beta_cond_60x20  - beta to BTC * BTC 20d ret (crypto conditioning)
  hilo_range_10x60     - 10d high-low range / 60d high-low range
  vol_ratio_5x60       - 5d vol / 60d vol (short vol spike detector)
  updown_vol_20        - vol of up days / vol of down days (asymmetry)
  gain_loss_ratio_20   - mean up-day ret / mean |down-day ret|
  xau_affinity_60      - 60d corr with XAU ret (safe-haven affinity)
  vix_affinity_60      - -60d corr with VIX change (risk-off affinity)
  trend_slope_accel_20x60 - (20d slope - 60d slope)/60d vol (acceleration)
  wti_rs_20            - asset 20d ret minus WTI 20d ret (energy RS)
  vix_beta_cond_60x20  - round-23 refresh: beta to VIX * VIX 20d change
  us10y_beta_cond_60x20- round-23 refresh: beta to US10Y * US10Y 20d ret
  vol_ratio_10x60      - round-23 refresh: 10d vol / 60d vol
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


# ---------------- round-24 new candidates ----------------
def cand_rsi_14(close):
    def rsi(s):
        d = s.diff()
        up = d.clip(lower=0.0).rolling(14, min_periods=8).mean()
        dn = (-d.clip(upper=0.0)).rolling(14, min_periods=8).mean()
        rs = up / (dn + 1e-12)
        return 100.0 - 100.0 / (1.0 + rs)
    return per_asset(rsi)(close)


def cand_trend_r2_20(close):
    def r2(s):
        x = np.arange(20, dtype=float)
        yy = np.log(s.values)
        out = np.full(len(s), np.nan)
        for i in range(19, len(s)):
            w = yy[i - 19:i + 1]
            if np.isnan(w).any():
                continue
            b = np.polyfit(x, w, 1)
            pred = np.polyval(b, x)
            ss_res = float(np.sum((w - pred) ** 2))
            ss_tot = float(np.sum((w - w.mean()) ** 2))
            out[i] = 1.0 - ss_res / (ss_tot + 1e-12)
        return pd.Series(out, index=s.index)
    return per_asset(r2)(close)


def cand_dd_depth_20(close):
    m20 = close.rolling(20, min_periods=10).max()
    return close / m20 - 1.0


def cand_rel_mom_60d_skip5(close):
    m60 = per_asset(lambda s: s.shift(5) / s.shift(65) - 1.0)(close)
    return m60.sub(m60.median(axis=1), axis=0)


def cand_btc_rs_20(close):
    btc = close["BTC"].dropna()
    btc20 = (btc / btc.shift(20) - 1.0).reindex(close.index).ffill()
    m20 = close / close.shift(20) - 1.0
    return m20 - btc20


def cand_btc_beta_cond_60x20(close):
    btc = close["BTC"].dropna()
    btc_r = btc.pct_change()
    btc20 = (btc / btc.shift(20) - 1.0)
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), btc_r.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        out[a] = (beta * btc20.reindex(s.index)).reindex(close.index)
    return out


def cand_hilo_range_10x60(close):
    hi60 = close.rolling(60, min_periods=30).max()
    lo60 = close.rolling(60, min_periods=30).min()
    hi10 = close.rolling(10, min_periods=6).max()
    lo10 = close.rolling(10, min_periods=6).min()
    return (hi10 - lo10) / (hi60 - lo60).replace(0, np.nan)


def cand_vol_ratio_5x60(close):
    r = close.pct_change()
    v5 = r.rolling(5, min_periods=3).std()
    v60 = r.rolling(60, min_periods=30).std()
    return v5 / (v60 + 1e-9)


def cand_updown_vol_20(close):
    r = close.pct_change()
    up = r.where(r > 0, np.nan)
    dn = r.where(r < 0, np.nan)
    upv = up.rolling(20, min_periods=6).std()
    dnv = dn.rolling(20, min_periods=6).std()
    return upv / (dnv + 1e-9)


def cand_gain_loss_ratio_20(close):
    r = close.pct_change()
    up = r.clip(lower=0.0).rolling(20, min_periods=10).mean()
    dn = (-r.clip(upper=0.0)).rolling(20, min_periods=10).mean()
    return up / (dn + 1e-9)


def cand_xau_affinity_60(close):
    r = close.pct_change()
    xau_r = r["XAU"]
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = r[a].rolling(60, min_periods=30).corr(xau_r)
    return out


def cand_vix_affinity_60(close, macro):
    vix = macro["VIX"].dropna()
    vix_chg = vix.diff()
    r = close.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        out[a] = -r[a].rolling(60, min_periods=30).corr(vix_chg.reindex(r.index))
    return out


def cand_trend_slope_accel_20x60(close):
    def slope(s, n):
        x = np.arange(n, dtype=float)
        yy = np.log(s.values)
        out = np.full(len(s), np.nan)
        for i in range(n - 1, len(s)):
            w = yy[i - n + 1:i + 1]
            if np.isnan(w).any():
                continue
            out[i] = float(np.polyfit(x, w, 1)[0])
        return pd.Series(out, index=s.index)
    r = close.pct_change()
    v60 = r.rolling(60, min_periods=30).std()
    s20 = per_asset(lambda s: slope(s, 20))(close)
    s60 = per_asset(lambda s: slope(s, 60))(close)
    return (s20 - s60) / (v60 + 1e-9)


def cand_wti_rs_20(close):
    wti = close["WTI"].dropna()
    wti20 = (wti / wti.shift(20) - 1.0).reindex(close.index).ffill()
    m20 = close / close.shift(20) - 1.0
    return m20 - wti20


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


def cand_vol_ratio_10x60(close):
    r = close.pct_change()
    v10 = r.rolling(10, min_periods=6).std()
    v60 = r.rolling(60, min_periods=30).std()
    return v10 / (v60 + 1e-9)


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

    print("##### B) CANDIDATE SCREENS (round 24 new + round 23 refresh) #####")
    cands = {
        "rsi_14": lambda: cand_rsi_14(close),
        "trend_r2_20": lambda: cand_trend_r2_20(close),
        "dd_depth_20": lambda: cand_dd_depth_20(close),
        "rel_mom_60d_skip5": lambda: cand_rel_mom_60d_skip5(close),
        "btc_rs_20": lambda: cand_btc_rs_20(close),
        "btc_beta_cond_60x20": lambda: cand_btc_beta_cond_60x20(close),
        "hilo_range_10x60": lambda: cand_hilo_range_10x60(close),
        "vol_ratio_5x60": lambda: cand_vol_ratio_5x60(close),
        "updown_vol_20": lambda: cand_updown_vol_20(close),
        "gain_loss_ratio_20": lambda: cand_gain_loss_ratio_20(close),
        "xau_affinity_60": lambda: cand_xau_affinity_60(close),
        "vix_affinity_60": lambda: cand_vix_affinity_60(close, macro),
        "trend_slope_accel_20x60": lambda: cand_trend_slope_accel_20x60(close),
        "wti_rs_20": lambda: cand_wti_rs_20(close),
        "vix_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "VIX"),
        "us10y_beta_cond_60x20": lambda: cand_macro_beta_cond(close, macro, "US10Y"),
        "vol_ratio_10x60": lambda: cand_vol_ratio_10x60(close),
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
