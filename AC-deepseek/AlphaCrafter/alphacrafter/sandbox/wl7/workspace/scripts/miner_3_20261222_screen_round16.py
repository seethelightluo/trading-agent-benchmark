"""
miner_3 cycle 2026-12-22: screen round 16.
Re-validates the 8-factor library (warm-up + live drift through 2026-12-21)
and screens fresh round-16 candidates. Admission gates (same as prior cycles):
|IC_h10| >= 0.007, |ICIR_h10| >= 0.084, max_abs_library_correlation < 0.5.

Round-16 candidates (fresh ideas, not previously screened in rounds 1-15):
  usdcny_beta_cond_60x20  - beta to USDCNY * CNY 20d trend (novel FX beta)
  skew_60d_skip5          - 60d return skewness, 5d skip (longer lottery horizon)
  downside_mag_ratio_20   - mean(|r||r<0) / mean(|r||r>0) over 20d (asym magnitude)
  volume_stability_20     - -(std(vol)/mean(vol), 20d) (low volume variance favored)
  days_since_high_60      - -(days since 60d rolling max) (recent peak favored)
  csi300_beta_cond_60x20  - beta to 000300.SH * China equity 20d trend (regional)
  wti_beta_cond_60x20     - beta to WTI * oil 20d trend (commodity-specific)
  vix_mom_cond_20x60      - demeaned 20d mom * sign(VIX 20d chg) (regime-cond mom)
  kurt_60d_skip5          - 60d kurtosis, 5d skip (longer tail window)
  path_efficiency_20      - |mom20| / sum(|r|,20) (trend path efficiency)
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


# ---------------- round-16 candidates ----------------
def cand_usdcny_beta_cond_60x20(close, macro):
    ref = macro["USDCNY"].dropna()
    ref20 = (ref / ref.shift(20) - 1.0)
    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        return -beta * ref20.reindex(s.index)  # CNY strengthening (USDCNY down) favored
    return per_asset(f)(close)


def cand_skew_60d_skip5(close):
    def f(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(60, min_periods=30).skew()
    return per_asset(f)(close)


def cand_downside_mag_ratio_20(close):
    def f(s):
        rr = s.pct_change()
        dn = rr.where(rr < 0, np.nan).abs()
        up = rr.where(rr > 0, np.nan).abs()
        return dn.rolling(20).mean() / up.rolling(20).mean()  # low = upside-heavy
    return per_asset(f)(close)


def cand_volume_stability_20(vol):
    def f(s):
        s = s.replace(0, np.nan)
        m = s.rolling(20).mean()
        sd = s.rolling(20).std()
        return -(sd / m)  # stable volume favored
    return per_asset(f)(vol)


def cand_days_since_high_60(close):
    def f(s):
        roll_max = s.rolling(60, min_periods=30).max()
        is_high = (s >= roll_max)
        out = []
        days_since = np.nan
        for v in is_high.values:
            if v:
                days_since = 0
            elif not np.isnan(days_since):
                days_since += 1
            out.append(days_since)
        return pd.Series(out, index=s.index).shift(1) * -1.0  # recent peak favored
    return per_asset(f)(close)


def cand_csi300_beta_cond_60x20(close):
    ref = close["000300.SH"].dropna()
    ref20 = (ref / ref.shift(20) - 1.0)
    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        return beta * ref20.reindex(s.index)
    return per_asset(f)(close)


def cand_wti_beta_cond_60x20(close):
    ref = close["WTI"].dropna()
    ref20 = (ref / ref.shift(20) - 1.0)
    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        return beta * ref20.reindex(s.index)
    return per_asset(f)(close)


def cand_vix_mom_cond_20x60(close, macro):
    vix = macro["VIX"].dropna()
    vix_chg = np.sign(vix / vix.shift(20) - 1.0).reindex(close.index).ffill()
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    demeaned = m20.sub(m20.median(axis=1), axis=0)
    return demeaned.mul(vix_chg, axis=0)


def cand_kurt_60d_skip5(close):
    def f(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(60, min_periods=30).kurt()
    return per_asset(f)(close)


def cand_path_efficiency_20(close):
    def f(s):
        rr = s.pct_change().abs()
        total = rr.rolling(20).sum()
        net = (s / s.shift(20) - 1.0).abs()
        return net / total.replace(0, np.nan)
    return per_asset(f)(close)


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
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets, warm-up through {WARM_END}, data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}\n")

    print("##### A) LIBRARY RE-VALIDATION (warm-up) #####")
    lib_results = {}
    for fid, sig in libsig.items():
        lib_results[fid] = validate(f"[LIB] {fid}", sig, close, libsig)

    print("##### A2) LIVE-WINDOW DRIFT (informational: 2026-07-16..2026-12-21) #####")
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

    print("##### B) CANDIDATE SCREENS (round 16) #####")
    cands = {
        "usdcny_beta_cond_60x20": lambda: cand_usdcny_beta_cond_60x20(close, macro),
        "skew_60d_skip5": lambda: cand_skew_60d_skip5(close),
        "downside_mag_ratio_20": lambda: cand_downside_mag_ratio_20(close),
        "volume_stability_20": lambda: cand_volume_stability_20(vol),
        "days_since_high_60": lambda: cand_days_since_high_60(close),
        "csi300_beta_cond_60x20": lambda: cand_csi300_beta_cond_60x20(close),
        "wti_beta_cond_60x20": lambda: cand_wti_beta_cond_60x20(close),
        "vix_mom_cond_20x60": lambda: cand_vix_mom_cond_20x60(close, macro),
        "kurt_60d_skip5": lambda: cand_kurt_60d_skip5(close),
        "path_efficiency_20": lambda: cand_path_efficiency_20(close),
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
