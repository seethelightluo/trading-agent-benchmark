"""miner_3 screening: round 11 - range/price-location, efficiency, autocorrelation,
beta-to-benchmark, stochastic/CCI oscillators, trend-acceleration, gap/shadow, vol-term-structure.

Round-10 feedback: all reversal/oscillator/seasonality/macro-corr candidates failed
(best macro-corr low IC; RSI/CMO failed gate). This round targets families built on
close + open/high/low + macro with distinct information content vs the 11-factor library:
  1. kaufman_eff_20   : Kaufman efficiency ratio (trend purity, directionless)
  2. close_loc_20     : avg close location inside daily range (intraday pattern)
  3. intraday_mom_20  : avg open-to-close return (intraday momentum)
  4. autocorr_20      : 20d lag-1 return autocorrelation (trend persistence)
  5. spx_beta_60      : rolling beta vs SPX (equity beta regime)
  6. xau_beta_60      : rolling beta vs XAU (safe-haven beta)
  7. btc_beta_60      : rolling beta vs BTC (risk-on beta)
  8. updown_ratio_60  : sum(up returns)/|sum(down returns)| over 60d
  9. stoch_14         : stochastic oscillator %K (14d)
 10. rsi_2            : Connors short RSI(2)
 11. cci_20           : commodity channel index (20d)
 12. trend_accel      : mom20 - mom60 (trend acceleration)
 13. gap_5d           : avg open-vs-prev-close gap over 5d
 14. shadow_ratio_20  : upper/(upper+lower) wick share averaged 20d
 15. range_60         : avg (high-low)/close over 60d (range vol)
 16. vol_ratio_10_60  : vol(10)/vol(60) term structure
 17. skew_60          : 60d return skewness (round-9 skew_20 was close: ICIR=0.077)
 18. breakout_20      : # days close > prior-20d max over last 20d

Admission gate h=10 (warm-up 2020-01-01..2026-07-15): |IC|>=0.007, |ICIR|>=0.084,
max_abs_library_correlation < 0.5 against reconstructed library.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS = 8
WARM_END = "2026-07-15"
DAYS = 4000


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
    """Vectorized cross-sectional Spearman IC per date (min MIN_ASSETS)."""
    f = factor.stack(dropna=False).rename("f")
    r = fwd.stack(dropna=False).rename("r")
    j = pd.concat([f, r], axis=1).dropna()
    if len(j) == 0:
        return pd.Series(dtype=float)
    j["fr"] = j.groupby(level=0)["f"].rank()
    j["rr"] = j.groupby(level=0)["r"].rank()
    cnt = j.groupby(level=0).size()
    keep = cnt[cnt >= MIN_ASSETS].index
    j = j[j.index.get_level_values(0).isin(keep)]
    g = j.groupby(level=0)[["fr", "rr"]]
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


def library_signals(close, high, low, vol, macro):
    lib = {}
    r = close.pct_change()
    lib["amihud_20"] = (r.abs() / vol).rolling(20).mean()
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        er = ew_r.reindex(s.index)
        z = pd.concat([s.pct_change().rename("r"), er.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = pd.DataFrame(cols, index=close.index)

    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(20).mean())
        tot = rr.rolling(20).std()
        return -(ds / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)(close)
    lib["max_ret_20d"] = r.rolling(20).max()
    lib["mom_10d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(15) - 1.0)(close)
    lib["mom_120d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(125) - 1.0)(close)
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    lib["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    vix = macro["VIX"].dropna()
    vix20 = (vix / vix.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), vix.pct_change().reindex(s.index).rename("v")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
        cols[a] = (-beta * vix20.reindex(s.index))
    lib["vix_beta_cond_60x20"] = pd.DataFrame(cols, index=close.index)
    lib["vol_adj_mom_20x60"] = per_asset(
        lambda s: (s.shift(5) / s.shift(25) - 1.0) / s.pct_change().rolling(60).std())(close)
    lib["vol_of_vol20x60"] = r.rolling(20).std().rolling(60).std()
    eur = macro["EURUSD"].dropna()
    eur20 = (eur / eur.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), eur.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (beta * eur20.reindex(s.index))
    lib["eurusd_beta_cond_60x20"] = pd.DataFrame(cols, index=close.index)
    return lib


# ---------------- candidates ----------------
def cand_kaufman_eff(close, n=20):
    def f(s):
        chg = (s - s.shift(n)).abs()
        vol = s.diff().abs().rolling(n).sum()
        return chg / vol.replace(0, np.nan)
    return per_asset(f)(close)


def cand_close_loc(close, high, low, n=20):
    rng = (high - low).replace(0, np.nan)
    loc = (close - low) / rng
    return loc.rolling(n).mean()


def cand_intraday_mom(close, open_, n=20):
    im = close / open_ - 1.0
    return im.rolling(n).mean()


def cand_autocorr(close, n=20):
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        rr = s.pct_change()
        z = pd.concat([rr.rename("r"), rr.shift(1).rename("l")], axis=1)
        ac = z["r"].rolling(n).cov(z["l"]) / z["r"].rolling(n).var()
        cols[a] = ac
    return pd.DataFrame(cols, index=close.index)


def bench_beta(close, bench, n=60):
    b = close[bench].dropna()
    br = b.pct_change()
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), br.reindex(s.index).rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(n).cov(z["m"]) / z["m"].rolling(n).var()
    return pd.DataFrame(cols, index=close.index)


def cand_updown_ratio(close, n=60):
    def f(s):
        rr = s.pct_change()
        up = rr.clip(lower=0).rolling(n).sum()
        dn = (-rr).clip(lower=0).rolling(n).sum()
        return up / dn.replace(0, np.nan)
    return per_asset(f)(close)


def cand_stoch(close, high, low, n=14):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return (close - ll) / (hh - ll).replace(0, np.nan)


def wilder_rsi(s, n=14):
    rr = s.pct_change()
    gain = rr.clip(lower=0.0)
    loss = (-rr).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def cand_cci(close, high, low, n=20):
    tp = (high + low + close) / 3.0
    def f(s):
        sma = s.rolling(n).mean()
        md = s.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (s - sma) / (0.015 * md.replace(0, np.nan))
    return per_asset(f)(tp)


def cand_trend_accel(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    m60 = per_asset(lambda s: s.shift(5) / s.shift(65) - 1.0)(close)
    return m20 - m60


def cand_gap_5d(close, open_, n=5):
    gap = open_ / close.shift(1) - 1.0
    return gap.rolling(n).mean()


def cand_shadow_ratio(close, open_, high, low, n=20):
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    ratio = upper / (upper + lower).replace(0, np.nan)
    return ratio.rolling(n).mean()


def cand_range_60(close, high, low, n=60):
    rng = (high - low) / close
    return rng.rolling(n).mean()


def cand_vol_ratio_10_60(close):
    r = close.pct_change()
    return r.rolling(10).std() / r.rolling(60).std()


def cand_skew_60(close, n=60):
    return per_asset(lambda s: s.pct_change().rolling(n).skew())(close)


def cand_breakout_20(close, n=20):
    prior_max = close.rolling(n).max().shift(1)
    return (close > prior_max).astype(float).rolling(n).mean()


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


def validate(name, factor, close, libsig):
    res = {"n_dates": int(factor.loc[:WARM_END].shape[0])}
    fwd10 = fwd_returns(close, 10)
    ic = rank_ic_series(factor.loc[:WARM_END], fwd10)
    direction = 1.0 if ic.mean() >= 0 else -1.0
    res["ic_h10"] = float(direction * ic.mean())
    res["icir_h10"] = float(direction * ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic > 0).mean()) if len(ic) else float("nan")
    res["n_h10"] = len(ic)
    res["decay"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        ic_h = rank_ic_series(factor.loc[:WARM_END], fwd_returns(close, h))
        res["decay"][str(h)] = float(direction * ic_h.mean()) if len(ic_h) else float("nan")
    valid = factor.loc[:WARM_END].notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(factor.loc[:WARM_END])
    corrs = stacked_corr(factor.loc[:WARM_END], {k: v for k, v in libsig.items()})
    res["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))}
    gate = abs(res["ic_h10"]) >= 0.007 and abs(res["icir_h10"]) >= 0.084
    lowcorr = res["max_abs_library_correlation"] < 0.5
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
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets, warm-up through {WARM_END}")
    print(f"open/high/low NaN frac: {open_.isna().mean().mean():.4f} / {high.isna().mean().mean():.4f} / {low.isna().mean().mean():.4f}")
    print(f"volume NaN frac: {vol.isna().mean().mean():.4f}; zero frac: {(vol==0).mean().mean():.4f}")

    cands = {
        "kaufman_eff_20": lambda: cand_kaufman_eff(close, 20),
        "close_loc_20": lambda: cand_close_loc(close, high, low, 20),
        "intraday_mom_20": lambda: cand_intraday_mom(close, open_, 20),
        "autocorr_20": lambda: cand_autocorr(close, 20),
        "spx_beta_60": lambda: bench_beta(close, "SPX", 60),
        "xau_beta_60": lambda: bench_beta(close, "XAU", 60),
        "btc_beta_60": lambda: bench_beta(close, "BTC", 60),
        "updown_ratio_60": lambda: cand_updown_ratio(close, 60),
        "stoch_14": lambda: cand_stoch(close, high, low, 14),
        "rsi_2": lambda: per_asset(lambda s: wilder_rsi(s, 2))(close),
        "cci_20": lambda: cand_cci(close, high, low, 20),
        "trend_accel": lambda: cand_trend_accel(close),
        "gap_5d": lambda: cand_gap_5d(close, open_, 5),
        "shadow_ratio_20": lambda: cand_shadow_ratio(close, open_, high, low, 20),
        "range_60": lambda: cand_range_60(close, high, low, 60),
        "vol_ratio_10_60": lambda: cand_vol_ratio_10_60(close),
        "skew_60": lambda: cand_skew_60(close, 60),
        "breakout_20": lambda: cand_breakout_20(close, 20),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===")
    print("\n===== SUMMARY =====")
    for name, r in results.items():
        print(f"{name:<18} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} cov={r['coverage_dates_ge8']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
