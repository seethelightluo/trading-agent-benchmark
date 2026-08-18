"""
miner_3 cycle 2028-06-06: screen round 31 (data thru 2028-06-05).
Context: last live block 2028-05-09->05-23 PnL +0.37% (screener cycle 31); regime
as of 06-05: VIX 44.09 (+67%/20d) SEVERE risk-off, DXY 103.04 (+2.8%/20d) USD
strength, USDJPY 165.72 (+2.5%/20d), EURUSD 1.12 (-2.9%/20d). Frozen feeds
NDX/SOX/000688/CN10Y persist (~28% dead weight). Round-30 (05-09) found
btc_beta_cond_60x20 PASS (IC=+0.0394 ICIR=+0.0998 maxcorr=0.405) but persistence
was never completed; re-test and persist here. days_since_high_60_neg (round-29
PASS) and overnight_share_20 errored in round-30 execution -> fixed re-test.

Round-31 candidates (regime-motivated):
  - days_since_high_60_neg  : round-29 PASS re-test (fixed)
  - overnight_share_20      : overnight gap share (fixed)
  - usdjpy_beta_cond_60x20  : USDJPY beta * USDJPY 20d mom (yen-carry regime)
  - copper_beta_cond_60x20  : COPPER beta * COPPER 20d mom (commodity cycle)
  - wti_beta_cond_60x20     : WTI beta * WTI 20d mom (energy cycle)
  - vol_ratio_20_60         : short/long realized vol (vol compression)
  - trend_snr_20            : |mom20|/vol20 trend signal-to-noise
  - sharpe_60d_skip5        : risk-adjusted 60d momentum (skip5)
  - rel_vol_20              : asset vol20 / EW-median vol20 (idiosyncratic vol)
  - btc_beta_cond_60x20     : round-30 PASS re-test + persistence

Also re-validate the 8 currently EFFECTIVE library factors on warm-up/live/recent
windows and report drift.

Admission gates (warm-up 2020-01-01..2026-07-15): |IC_h10| >= 0.007,
|ICIR_h10| >= 0.084, max_abs_library_correlation < 0.5.
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


# ---------------- round-31 candidates ----------------
def cand_days_since_high_60_neg(close):
    """Negative days since 60d high (breakout recency). FIXED raw apply."""
    def f(s):
        days = s.rolling(60, min_periods=30).apply(
            lambda x: np.argmax(x[::-1]) if len(x) else np.nan, raw=True)
        return -days
    return per_asset(f)(close)


def cand_overnight_share_20(close, open_):
    """Share of 20d total |return| coming from overnight gaps (open/prev_close). FIXED."""
    cols = {}
    for a in close.columns:
        s, o = close[a].dropna(), open_[a].dropna()
        s, o = s.align(o, join="inner")
        gap = np.log(o / s.shift(1)).fillna(0.0)
        intra = np.log(s / o).fillna(0.0)
        tot = gap + intra
        cols[a] = gap.rolling(20, min_periods=10).sum() / tot.rolling(20, min_periods=10).sum().abs()
    return pd.DataFrame(cols, index=close.index)


def cand_ref_beta_cond(close, ref, window=60, mom=20):
    """beta to ref * ref 20d momentum (conditional regime exposure)."""
    ref20 = (ref / ref.shift(mom) - 1.0)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(window).cov(z["x"]) / z["x"].rolling(window).var()
        return beta * ref20.reindex(s.index)
    return per_asset(f)(close)


def cand_vol_ratio_20_60(close):
    """Short/long realized vol: 20d vol / 60d vol (vol compression when <1)."""
    def f(s):
        rr = s.pct_change()
        return rr.rolling(20, min_periods=12).std() / rr.rolling(60, min_periods=30).std()
    return per_asset(f)(close)


def cand_trend_snr_20(close):
    """Trend signal-to-noise: |20d momentum| / 20d realized vol (skip 5 on mom)."""
    def f(s):
        mom = s.shift(5) / s.shift(25) - 1.0
        vol = s.pct_change().rolling(20, min_periods=12).std()
        return mom.abs() / vol
    return per_asset(f)(close)


def cand_sharpe_60d_skip5(close):
    """Risk-adjusted 60d momentum: mean/std of daily returns (skip 5), scaled."""
    def f(s):
        rr = s.pct_change().shift(5)
        m = rr.rolling(60, min_periods=30).mean()
        sd = rr.rolling(60, min_periods=30).std()
        return m / sd
    return per_asset(f)(close)


def cand_rel_vol_20(close):
    """Idiosyncratic vol: asset 20d vol / EW-median 20d vol."""
    vol = per_asset(lambda s: s.pct_change().rolling(20, min_periods=12).std())(close)
    med = vol.median(axis=1)
    return vol.div(med, axis=0)


def validate(name, factor, close, libsig, direction=1.0):
    res = {"name": name, "n_dates": int(factor.shape[0])}
    window_end = pd.Timestamp(WARM_END)
    fwd10 = fwd_returns(close, 10)
    ic10 = rank_ic_series(factor.loc[:window_end], fwd10)
    res["n_h10"] = len(ic10)
    res["ic_h10"] = float(direction * ic10.mean()) if len(ic10) else float("nan")
    res["icir_h10"] = float(direction * ic10.mean() / ic10.std()) if len(ic10) > 2 and ic10.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic10 > 0).mean()) if len(ic10) else float("nan")
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

    if factor.index[-1] > pd.Timestamp(LIVE_START):
        for tag, sl in (("live", slice(LIVE_START, None)), ("recent12m", slice(pd.Timestamp("2027-06-06"), None))):
            try:
                ic_l = rank_ic_series(factor.loc[sl], fwd10)
                res[f"{tag}_ic_h10"] = float(direction * ic_l.mean()) if len(ic_l) else float("nan")
                res[f"{tag}_icir_h10"] = float(direction * ic_l.mean() / ic_l.std()) if len(ic_l) > 2 and ic_l.std() > 0 else float("nan")
                res[f"{tag}_n"] = len(ic_l)
            except Exception:
                res[f"{tag}_ic_h10"], res[f"{tag}_icir_h10"], res[f"{tag}_n"] = float("nan"), float("nan"), 0

    print(f"=== {name} === dates={res['n_dates']} direction={direction:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  decay={res['decay']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    if "live_ic_h10" in res:
        print(f"  LIVE {LIVE_START}..{factor.index[-1].date()}: IC={res['live_ic_h10']:+.4f} ICIR={res['live_icir_h10']:+.4f} n={res['live_n']} | recent12m IC={res['recent12m_ic_h10']:+.4f} ICIR={res['recent12m_icir_h10']:+.4f} n={res['recent12m_n']}")
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}\n")

    print("regime sanity: ", end="")
    for s in ["VIX", "DXY", "EURUSD", "USDJPY", "USDCNY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%} ", end="")
    print("\n")

    # ---- re-validate library factors (drift report) ----
    print("##### LIBRARY RE-VALIDATION (drift) #####")
    fwd10 = fwd_returns(close, 10)
    for fid, sig in libsig.items():
        warm = sig.loc[:pd.Timestamp(WARM_END)]
        ic_w = rank_ic_series(warm, fwd10)
        icw = float(ic_w.mean()) if len(ic_w) else float("nan")
        icirw = float(ic_w.mean() / ic_w.std()) if len(ic_w) > 2 and ic_w.std() > 0 else float("nan")
        ic_l = rank_ic_series(sig.loc[LIVE_START:], fwd10)
        icl = float(ic_l.mean()) if len(ic_l) else float("nan")
        icirl = float(ic_l.mean() / ic_l.std()) if len(ic_l) > 2 and ic_l.std() > 0 else float("nan")
        ic_r = rank_ic_series(sig.loc["2027-06-06":], fwd10)
        icr = float(ic_r.mean()) if len(ic_r) else float("nan")
        icirr = float(ic_r.mean() / ic_r.std()) if len(ic_r) > 2 and ic_r.std() > 0 else float("nan")
        flag = "DRIFT!" if (icirr < 0 and abs(icr) < IC_GATE) or (icirl < 0 and abs(icl) < IC_GATE) else "ok"
        print(f"  {fid:26s} warm IC={icw:+.4f} ICIR={icirw:+.4f} n={len(ic_w):4d} | "
              f"live IC={icl:+.4f} ICIR={icirl:+.4f} n={len(ic_l):4d} | "
              f"recent12m IC={icr:+.4f} ICIR={icirr:+.4f} n={len(ic_r):4d} [{flag}]")
    print()

    # ---- candidates ----
    cands = {
        "days_since_high_60_neg": lambda: cand_days_since_high_60_neg(close),
        "overnight_share_20": lambda: cand_overnight_share_20(close, open_),
        "usdjpy_beta_cond_60x20": lambda: cand_ref_beta_cond(close, macro["USDJPY"].dropna()),
        "copper_beta_cond_60x20": lambda: cand_ref_beta_cond(close, close["COPPER"].dropna()),
        "wti_beta_cond_60x20": lambda: cand_ref_beta_cond(close, close["WTI"].dropna()),
        "vol_ratio_20_60": lambda: cand_vol_ratio_20_60(close),
        "trend_snr_20": lambda: cand_trend_snr_20(close),
        "sharpe_60d_skip5": lambda: cand_sharpe_60d_skip5(close),
        "rel_vol_20": lambda: cand_rel_vol_20(close),
        "btc_beta_cond_60x20": lambda: cand_ref_beta_cond(close, close["BTC"].dropna()),
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
        live = f" liveIC={res_['live_ic_h10']:+.4f}" if "live_ic_h10" in res_ else ""
        print(f"{name}: IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} hit={res_['hit_h10']:.3f} "
              f"cov_ge8={res_['coverage_dates_ge8']:.3f} turn={res_['turnover_10d_rank']:.3f} "
              f"maxcorr={res_['max_abs_library_correlation']:.3f}{live} "
              f"-> {'PASS' if res_['PASS'] else 'FAIL'}")
