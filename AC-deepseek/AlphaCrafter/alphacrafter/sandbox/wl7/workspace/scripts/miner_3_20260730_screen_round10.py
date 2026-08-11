"""miner_3 cycle 20 screening: round 10 - reversal / oscillator / seasonality / macro-corr families.

Round-9 feedback: all 10 candidates failed gate (best skew_20d ICIR=0.0768 < 0.084; several had
high library corr or NaN coverage due to zero volume on SOX/XAU/COPPER/WTI/US10Y/CN10Y).
This round targets families built on close + macro only (robust coverage):
  1. rev_1d / rev_3d / rev_5d / rev_10d : short-term reversal (negated past return)
  2. rsi_14 / rsi_5                    : Wilder RSI mean-reversion oscillator
  3. cmo_20                            : Chande momentum oscillator
  4. high_252                          : 52-week high proximity
  5. rebound_60                        : close / rolling_min(close,60) - 1
  6. winrate_20                        : fraction of positive days over 20d
  7. season_3y                         : same-calendar-month mean return over prior 3 years
  8. volreg_mom                        : rel_mom_20d_skip5 * (1 - VIX 60d percentile)
  9. rev5_vix                          : -ret_5d * VIX 60d percentile (vol-conditional reversal)
 10. dur_corr_60 / dxy_corr_60 / jpy_corr_60 : 60d rolling corr with US10Y / DXY / USDJPY
 11. macd_hist                         : (EMA12-EMA26) - EMA9(EMA12-EMA26)
 12. slope_60                          : OLS slope of log(close) vs time over 60d
Admission gate h=10 (warm-up 2020-01-01..2026-07-15): |IC|>=0.007, |ICIR|>=0.084,
max_abs_library_correlation < 0.5 against reconstructed 11-factor library.
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
    ics = []
    for d in factor.index.intersection(fwd.index):
        f = factor.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        if len(r) >= MIN_ASSETS:
            ics.append((d, r.corr(f.reindex(r.index), method="spearman")))
    return pd.Series(dict(ics)).sort_index()


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


# ---------------- candidates (per-asset calendar-aware) ----------------
def cand_rev_k(close, k):
    return per_asset(lambda s: -(s / s.shift(k) - 1.0))(close)


def wilder_rsi(s, n=14):
    rr = s.pct_change()
    gain = rr.clip(lower=0.0)
    loss = (-rr).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def cand_rsi(close, n):
    return per_asset(lambda s: wilder_rsi(s, n))(close)


def cand_cmo(close, n=20):
    def f(s):
        rr = s.pct_change()
        up = rr.clip(lower=0.0).rolling(n).sum()
        dn = (-rr).clip(lower=0.0).rolling(n).sum()
        return 100.0 * (up - dn) / (up + dn).replace(0, np.nan)
    return per_asset(f)(close)


def cand_high_252(close):
    return per_asset(lambda s: s / s.rolling(252).max() - 1.0)(close)


def cand_rebound_60(close):
    return per_asset(lambda s: s / s.rolling(60).min() - 1.0)(close)


def cand_winrate_20(close):
    return per_asset(lambda s: (s.pct_change() > 0).rolling(20).mean())(close)


def cand_season_3y(close):
    """same-calendar-month mean return over prior 3 calendar years (monthly-updated)."""
    out = {}
    for a in close.columns:
        s = close[a].dropna()
        rr = s.pct_change()
        yr = rr.index.year
        mo = rr.index.month
        monthly = pd.DataFrame({"y": yr, "m": mo, "r": rr.values}).dropna()
        agg = monthly.groupby(["y", "m"])["r"].mean().rename("mr").reset_index()
        agg = agg.sort_values(["y", "m"])
        season = {}
        for y in sorted(agg["y"].unique()):
            for m in range(1, 13):
                prior = agg[(agg["y"] < y) & (agg["m"] == m) & (agg["y"] >= y - 3)]
                season[(y, m)] = float(prior["mr"].mean()) if len(prior) else np.nan
        vals = np.full(len(rr), np.nan)
        for i, (dt, y, m) in enumerate(zip(rr.index, yr, mo)):
            v = season.get((y, m), np.nan)
            if np.isfinite(v):
                vals[i] = v
        out[a] = pd.Series(vals, index=rr.index)
    return pd.DataFrame(out, index=close.index)


def vix_percentile_60(macro):
    vix = macro["VIX"].dropna()
    pct = vix.rolling(60).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    return pct


def cand_volreg_mom(close, macro, vix_pct):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    rel = m20.sub(m20.median(axis=1), axis=0)
    return rel * (1.0 - vix_pct.reindex(close.index))


def cand_rev5_vix(close, vix_pct):
    rev5 = per_asset(lambda s: -(s / s.shift(5) - 1.0))(close)
    return rev5 * vix_pct.reindex(close.index)


def macro_corr(close, macro, name, win=60):
    m = macro[name].dropna()
    mr = m.pct_change()
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), mr.reindex(s.index).rename("x")], axis=1).dropna()
        cols[a] = z["r"].rolling(win).corr(z["x"])
    return pd.DataFrame(cols, index=close.index)


def cand_macd_hist(close):
    def f(s):
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return macd - macd.ewm(span=9, adjust=False).mean()
    return per_asset(f)(close)


def cand_slope_60(close):
    def f(s):
        x = np.arange(len(s), dtype=float)
        y = np.log(s.values)
        m = 60
        out = np.full(len(s), np.nan)
        for i in range(m - 1, len(s)):
            yy = y[i - m + 1:i + 1]
            xx = x[i - m + 1:i + 1]
            xx = xx - xx.mean()
            b = np.dot(xx, yy - yy.mean()) / np.dot(xx, xx)
            out[i] = b * m
        return pd.Series(out, index=s.index)
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
    print(f"library factors: {sorted(libsig.keys())}")

    vix_pct = vix_percentile_60(macro)
    cands = {
        "rev_1d": lambda: cand_rev_k(close, 1),
        "rev_3d": lambda: cand_rev_k(close, 3),
        "rev_5d": lambda: cand_rev_k(close, 5),
        "rev_10d": lambda: cand_rev_k(close, 10),
        "rsi_14": lambda: cand_rsi(close, 14),
        "rsi_5": lambda: cand_rsi(close, 5),
        "cmo_20": lambda: cand_cmo(close, 20),
        "high_252": lambda: cand_high_252(close),
        "rebound_60": lambda: cand_rebound_60(close),
        "winrate_20": lambda: cand_winrate_20(close),
        "season_3y": lambda: cand_season_3y(close),
        "volreg_mom": lambda: cand_volreg_mom(close, macro, vix_pct),
        "rev5_vix": lambda: cand_rev5_vix(close, vix_pct),
        "dur_corr_60": lambda: macro_corr(close, macro, "US10Y", 60),
        "dxy_corr_60": lambda: macro_corr(close, macro, "DXY", 60),
        "jpy_corr_60": lambda: macro_corr(close, macro, "USDJPY", 60),
        "macd_hist": lambda: cand_macd_hist(close),
        "slope_60": lambda: cand_slope_60(close),
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
        print(f"{name:<15} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} cov={r['coverage_dates_ge8']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
