"""miner_3 cycle 18 screening: new factor families + fixed updown_vol_20.

Candidates (all 15-asset cross-sectional, h=10 admission gate |IC|>=0.007,
|ICIR|>=0.084, max library corr < 0.5 against the reconstructed 10-factor library):
  1. updown_vol_20       : 20d mean volume on up-days / mean volume on down-days (fixed call)
  2. overnight_mom_20x5  : 20d momentum of overnight returns (open/prev_close-1), skip 5
  3. stoch_pos_20        : (close - min(low,20)) / (max(high,20) - min(low,20)), 20d %K position
  4. sharpe_60d          : rolling 60d mean/std of daily returns (risk-adjusted return)
  5. gold_lead_20        : beta(asset, XAU, 60) * (XAU/XAU.shift(20)-1)
  6. updown_ret_20       : mean up-day return / |mean down-day return| over 20d
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


def load_ohlcv(days=4000):
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
        df = get_index_daily_data(s, days=4000)
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
    # newly persisted factors are also part of the effective library now
    eur = macro["EURUSD"].dropna()
    eur20 = (eur / eur.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), eur.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (beta * eur20.reindex(s.index))
    lib["eurusd_beta_cond_60x20"] = pd.DataFrame(cols, index=close.index)
    btc = close["BTC"].dropna()
    btc20 = (btc / btc.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), btc.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (beta * btc20.reindex(s.index))
    lib["btc_lead_20"] = pd.DataFrame(cols, index=close.index)
    return lib


# ---------------- candidates ----------------
def cand_updown_vol_20(close, *_):
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        v = vol[a].reindex(s.index).dropna()
        r = s.pct_change().reindex(v.index)
        df = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
        upm = df["v"].where(df["r"] > 0).rolling(20).mean()
        dnm = df["v"].where(df["r"] < 0).rolling(20).mean()
        cols[a] = (upm / dnm)
    return pd.DataFrame(cols, index=close.index)


def cand_overnight_mom_20x5(close, open_, *_):
    on = open_ / close.shift(1) - 1.0
    return per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(on)


def cand_stoch_pos_20(close, open_, high, low, *_):
    cols = {}
    for a in close.columns:
        c, h, l = close[a].dropna(), high[a].dropna(), low[a].dropna()
        hi = h.rolling(20).max()
        lo = l.rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        cols[a] = ((c - lo) / rng).clip(0, 1)
    return pd.DataFrame(cols, index=close.index)


def cand_sharpe_60d(close, *_):
    return per_asset(lambda s: s.pct_change().rolling(60).mean() / s.pct_change().rolling(60).std())(close)


def cand_gold_lead_20(close, *_):
    lead = close["XAU"].dropna()
    lm = (lead / lead.shift(20) - 1.0)
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), lead.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (beta * lm.reindex(s.index))
    return pd.DataFrame(cols, index=close.index)


def cand_updown_ret_20(close, *_):
    def f(s):
        rr = s.pct_change()
        up = rr.where(rr > 0).rolling(20).mean()
        dn = rr.where(rr < 0).rolling(20).mean()
        return up / dn.abs()
    return per_asset(f)(close)


# ---------------- validation ----------------
def stacked_corr(cand, libsig):
    out = {}
    f = cand.stack().rename("f")
    for fid, ls in libsig.items():
        both = pd.concat([f, ls.stack().rename("l")], axis=1).dropna()
        if len(both) >= 200:
            out[fid] = float(np.corrcoef(both["f"], both["l"])[0, 1])
    return out


def validate(name, factor, close, libsig):
    factor_w = factor.loc[:WARM_END]
    res = {"name": name, "n_dates": len(factor_w)}
    fwd = {h: fwd_returns(close, h) for h in (1, 2, 3, 5, 10, 20)}
    ics = {h: rank_ic_series(factor_w, fwd[h]) for h in fwd}
    ic10 = ics[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    ics = {h: ic * direction for h, ic in ics.items()}
    for h in (1, 2, 3, 5, 10, 20):
        ic = ics[h]
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
        res[f"n_h{h}"] = len(ic)
    res["direction"] = direction
    valid = factor_w.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(factor_w)
    corrs = stacked_corr(factor_w, libsig)
    res["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))}
    res["decay"] = {str(h): round(res[f"ic_h{h}"], 4) for h in (1, 2, 3, 5, 10, 20)}
    gate = abs(res["ic_h10"]) >= 0.007 and abs(res["icir_h10"]) >= 0.084
    lowcorr = res["max_abs_library_correlation"] < 0.5
    res["PASS"] = gate and lowcorr
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

    cands = {
        "updown_vol_20": lambda: cand_updown_vol_20(close, open_, high, low, vol),
        "overnight_mom_20x5": lambda: cand_overnight_mom_20x5(close, open_, high, low, vol),
        "stoch_pos_20": lambda: cand_stoch_pos_20(close, open_, high, low, vol),
        "sharpe_60d": lambda: cand_sharpe_60d(close, open_, high, low, vol),
        "gold_lead_20": lambda: cand_gold_lead_20(close, open_, high, low, vol),
        "updown_ret_20": lambda: cand_updown_ret_20(close, open_, high, low, vol),
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
        print(f"{name:<22} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} -> {'PASS' if r['PASS'] else 'FAIL'}")
