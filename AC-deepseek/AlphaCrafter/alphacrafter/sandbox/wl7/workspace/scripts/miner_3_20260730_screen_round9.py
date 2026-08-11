"""miner_3 cycle 19 screening: orthogonal higher-moment / efficiency / cross-asset lead families.

Round-8 feedback: stoch_pos_20 passed IC gate but corr=0.724 (downside_vol_ratio); sharpe_60d
corr=0.598 (vol_adj_mom); updown_vol_20 & updown_ret_20 had NaN coverage. This round targets
families far from the existing 11-factor library:
  1. skew_20d        : 20d rolling skewness of daily returns (higher moment)
  2. kurt_60d        : 60d rolling excess kurtosis (tail shape)
  3. autocorr_10d    : 10d lag-1 autocorrelation of daily returns (trend vs reversal)
  4. eff_ratio_20    : Kaufman efficiency |C-C[-20]| / sum|ret| over 20d
  5. vol_z_60        : volume / rolling-60d mean volume (abnormal volume)
  6. overnight_share_20 : 20d mean |open/prev_close-1| / (|overnight|+|intraday|)
  7. vol_term_20x60  : 20d vol / 60d vol (volatility term slope)
  8. copper_lead_20  : beta(asset,COPPER,60) * COPPER 20d return
  9. ndx_lead_20     : beta(asset,NDX,60) * NDX 20d return
 10. maxdd_60        : close/rolling_max(close,60)-1 (drawdown)
Admission gate h=10: |IC|>=0.007, |ICIR|>=0.084, max abs library corr < 0.5.
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


# ---------------- candidates ----------------
def cand_skew_20d(close, *_):
    return per_asset(lambda s: s.pct_change().rolling(20).skew())(close)


def cand_kurt_60d(close, *_):
    return per_asset(lambda s: s.pct_change().rolling(60).kurt())(close)


def cand_autocorr_10d(close, *_):
    def f(s):
        rr = s.pct_change()
        return rr.rolling(10).corr(rr.shift(1))
    return per_asset(f)(close)


def cand_eff_ratio_20(close, *_):
    def f(s):
        r = s.pct_change().abs()
        total = r.rolling(20).sum()
        net = (s - s.shift(20)).abs()
        return net / total.replace(0, np.nan)
    return per_asset(f)(close)


def cand_vol_z_60(close, open_, high, low, vol):
    m = vol.rolling(60).mean()
    return (vol / m.replace(0, np.nan)).clip(0, 10)


def cand_overnight_share_20(close, open_, *_):
    on = (open_ / close.shift(1) - 1.0).abs()
    intra = (close / open_ - 1.0).abs()
    num = on.rolling(20).mean()
    den = (on + intra).rolling(20).mean()
    return (num / den.replace(0, np.nan))


def cand_vol_term_20x60(close, *_):
    def f(s):
        rr = s.pct_change()
        return rr.rolling(20).std() / rr.rolling(60).std().replace(0, np.nan)
    return per_asset(f)(close)


def lead_factor(close, lead_name, win=60, lead_win=20):
    lead = close[lead_name].dropna()
    lm = (lead / lead.shift(lead_win) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), lead.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(win).cov(z["x"]) / z["x"].rolling(win).var().replace(0, np.nan)
        cols[a] = (beta * lm.reindex(s.index))
    return pd.DataFrame(cols, index=close.index)


def cand_copper_lead_20(close, *_):
    return lead_factor(close, "COPPER")


def cand_ndx_lead_20(close, *_):
    return lead_factor(close, "NDX")


def cand_maxdd_60(close, *_):
    return per_asset(lambda s: s / s.rolling(60).max() - 1.0)(close)


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

    cands = {
        "skew_20d": lambda: cand_skew_20d(close, open_, high, low, vol),
        "kurt_60d": lambda: cand_kurt_60d(close, open_, high, low, vol),
        "autocorr_10d": lambda: cand_autocorr_10d(close, open_, high, low, vol),
        "eff_ratio_20": lambda: cand_eff_ratio_20(close, open_, high, low, vol),
        "vol_z_60": lambda: cand_vol_z_60(close, open_, high, low, vol),
        "overnight_share_20": lambda: cand_overnight_share_20(close, open_, high, low, vol),
        "vol_term_20x60": lambda: cand_vol_term_20x60(close, open_, high, low, vol),
        "copper_lead_20": lambda: cand_copper_lead_20(close, open_, high, low, vol),
        "ndx_lead_20": lambda: cand_ndx_lead_20(close, open_, high, low, vol),
        "maxdd_60": lambda: cand_maxdd_60(close, open_, high, low, vol),
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
        print(f"{name:<20} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} cov={r['coverage_dates_ge8']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
