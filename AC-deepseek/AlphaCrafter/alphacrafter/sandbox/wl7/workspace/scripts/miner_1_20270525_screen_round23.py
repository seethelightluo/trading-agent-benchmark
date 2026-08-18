"""
miner_1 cycle 2027-05-25: screen round 23 (fresh factor families).
Universe: 15 tradable cross-asset instruments (2020-01-01..2026-07-15 warm-up
admission; live drift informational 2026-07-16..2027-05-24).
IC = cross-sectional Spearman rank IC per date (>=8 assets).
Admission gates (benchmark contract): |IC_h10|>=0.007, |ICIR_h10|>=0.084,
max_abs_library_correlation < 0.5 vs the CURRENT 7-factor effective library.

Round-23 candidates (fresh families not screened in rounds 1-22):
  bear_beta_60            - 60d beta estimated on market-down days only (downside exposure)
  bear_bull_ratio_60      - bear beta / bull beta (downside vs upside sensitivity)
  coskew_60               - 60d coskewness vs EW market (systematic tail-risk loading)
  idio_skew_20_skip5      - 20d skewness of EW-regression residuals, skip5 (idio tail shape)
  mom_ewma_20_skip5       - recency-weighted EWMA trend of daily returns (halflife 10), skip5
  spread_beta_cond_60x20  - beta to (US10Y-CN10Y) spread changes x spread momentum
  calmar_60x20_skip5      - 60d return / 60d max drawdown, skip5
  updown_vol_ratio_20_skip5 - 20d up-day vol / down-day vol, skip5 (vol asymmetry)
  rev_5d_skip1            - 5d short-horizon reversal, skip1
  mom_tail_cond_20x60     - 20d momentum x sign(60d skewness), skip5
  leverage_effect_20      - 20d corr(r_t, |r_{t-1}|), skip5 (vol feedback)
  beta_trend_20x60        - change in 60d EW beta over 20d, skip5 (beta drift)
  xau_dd_rel_20_skip5     - asset 20d drawdown minus XAU 20d drawdown, skip5 (risk-off relative)
  cn10y_beta_cond_60x20   - beta to CN10Y returns x CN10Y momentum, skip5
"""
from __future__ import annotations
import io, json, zlib, base64
from pathlib import Path
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
HORIZONS = (1, 2, 3, 5, 10, 20)
EPS = 1e-12

LIB_FACTORS = ["rel_mom_20d_skip5", "beta_ew_60d", "downside_vol_ratio_20",
               "max_ret_20d", "eurusd_beta_cond_60x20", "corr_ew_60", "kurt_20d_skip5"]


def load_data():
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=DAYS)
        if df is None or not len(df):
            print("WARN no data", s)
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
    macro = {}
    for m in MACRO:
        df = get_index_daily_data(m, days=DAYS)
        if df is not None and len(df):
            macro[m] = df.set_index("date")["close"].astype(float)
    return _p(closes), _p(opens), _p(highs), _p(lows), _p(vols), macro


def fwd_returns(panel, h):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def row_pearson(X, Y, min_n=MIN_ASSETS):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    valid = np.isfinite(X) & np.isfinite(Y)
    cnt = valid.sum(axis=1)
    Xv = np.where(valid, X, np.nan)
    Yv = np.where(valid, Y, np.nan)
    Xc = Xv - np.nanmean(Xv, axis=1, keepdims=True)
    Yc = Yv - np.nanmean(Yv, axis=1, keepdims=True)
    num = np.nansum(Xc * Yc, axis=1)
    dx = np.sqrt(np.nansum(Xc * Xc, axis=1))
    dy = np.sqrt(np.nansum(Yc * Yc, axis=1))
    r = np.full(len(X), np.nan)
    m = (cnt >= min_n) & (dx > 0) & (dy > 0)
    r[m] = num[m] / (dx[m] * dy[m])
    return r


def rank_ic_series(factor, fwd):
    F = factor.rank(axis=1).values.astype(float)
    R = fwd.rank(axis=1).values.astype(float)
    return pd.Series(row_pearson(F, R), index=factor.index)


def turnover_10d_rank(factor):
    ranks = factor.rank(axis=1).values.astype(float)
    a, b = ranks[:-10], ranks[10:]
    valid = np.isfinite(a) & np.isfinite(b)
    cnt = valid.sum(axis=1)
    ok = cnt >= MIN_ASSETS
    m = np.full(len(a), np.nan)
    m[ok] = np.nansum(np.abs(a - b) * valid, axis=1)[ok] / cnt[ok]
    return float(np.nanmean(m))


def per_asset(fn):
    def wrapper(panel):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


# ---------------- current effective library (7 factors, screener definitions) ----------------
def library_signals(close, macro):
    r = close.pct_change()
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    lib = {}
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

    def eur_beta_cond(s):
        eur = macro["EURUSD"].pct_change()
        eur_mom = macro["EURUSD"] / macro["EURUSD"].shift(20) - 1.0
        z = pd.concat([s.pct_change().rename("r"), eur.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        return beta * eur_mom.reindex(s.index)
    lib["eurusd_beta_cond_60x20"] = per_asset(eur_beta_cond)(close)

    def kurt(s):
        rr = s.pct_change()
        return rr.rolling(20, min_periods=10).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close).shift(5)
    return lib


# ---------------- round-23 candidates ----------------
def _ew_beta_series(s, ew_r):
    z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
    return z["r"].rolling(60, min_periods=30).cov(z["m"]) / z["m"].rolling(60, min_periods=30).var()


def cand_bear_beta_60(close):
    ew_r = close.pct_change().mean(axis=1)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        down = z[z["m"] < 0]
        if len(down) < 30:
            return pd.Series(np.nan, index=z.index)
        beta = down["r"].rolling(60, min_periods=20).cov(down["m"]) / down["m"].rolling(60, min_periods=20).var()
        return beta
    return per_asset(f)(close).shift(5)


def cand_bear_bull_ratio_60(close):
    ew_r = close.pct_change().mean(axis=1)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        dn = z[z["m"] < 0]
        up = z[z["m"] > 0]
        if len(dn) < 20 or len(up) < 20:
            return pd.Series(np.nan, index=z.index)
        bb = dn["r"].rolling(60, min_periods=20).cov(dn["m"]) / dn["m"].rolling(60, min_periods=20).var()
        bu = up["r"].rolling(60, min_periods=20).cov(up["m"]) / up["m"].rolling(60, min_periods=20).var()
        return bb / (bu + EPS)
    return per_asset(f)(close).shift(5)


def cand_coskew_60(close):
    ew_r = close.pct_change().mean(axis=1)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        m2 = z["m"] ** 2
        num = z["r"].rolling(60, min_periods=30).cov(m2)
        den = z["r"].rolling(60, min_periods=30).std() * z["m"].rolling(60, min_periods=30).var()
        return num / (den + EPS)
    return per_asset(f)(close).shift(5)


def cand_idio_skew_20_skip5(close):
    ew_r = close.pct_change().mean(axis=1)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["m"]) / z["m"].rolling(60, min_periods=30).var()
        resid = z["r"] - beta * z["m"]
        return resid.rolling(20, min_periods=10).skew()
    return per_asset(f)(close).shift(5)


def cand_mom_ewma_20_skip5(close):
    r = close.pct_change()
    return r.ewm(halflife=10, min_periods=10).mean().shift(5)


def cand_spread_beta_cond_60x20(close):
    spread = close["US10Y"] - close["CN10Y"]
    sp_r = spread.pct_change()
    sp_mom = spread / spread.shift(20) - 1.0

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), sp_r.rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["x"]) / z["x"].rolling(60, min_periods=30).var()
        return beta * sp_mom.reindex(s.index)
    return per_asset(f)(close).shift(5)


def cand_calmar_60x20_skip5(close):
    ret60 = close / close.shift(60) - 1.0
    dd = close / close.rolling(60, min_periods=30).max() - 1.0
    mdd = dd.rolling(60, min_periods=30).min()
    return (ret60 / (EPS - mdd)).shift(5)


def cand_updown_vol_ratio_20_skip5(close):
    r = close.pct_change()
    up = r.where(r > 0, np.nan)
    dn = r.where(r < 0, np.nan)
    up_std = up.rolling(20, min_periods=10).std()
    dn_std = dn.rolling(20, min_periods=10).std()
    return (up_std / (dn_std + EPS)).shift(5)


def cand_rev_5d_skip1(close):
    return -(close.shift(1) / close.shift(6) - 1.0)


def cand_mom_tail_cond_20x60(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    sk60 = close.pct_change().rolling(60, min_periods=30).skew()
    return m20 * np.sign(sk60.shift(5))


def cand_leverage_effect_20(close):
    r = close.pct_change()
    return r.rolling(20, min_periods=10).corr(r.abs().shift(1)).shift(5)


def cand_beta_trend_20x60(close):
    ew_r = close.pct_change().mean(axis=1)
    b = per_asset(lambda s: _ew_beta_series(s, ew_r))(close)
    return (b - b.shift(20)).shift(5)


def cand_xau_dd_rel_20_skip5(close):
    dd = close / close.rolling(20, min_periods=10).max() - 1.0
    return (dd - dd["XAU"]).shift(5)


def cand_cn10y_beta_cond_60x20(close):
    cn_r = close["CN10Y"].pct_change()
    cn_mom = close["CN10Y"] / close["CN10Y"].shift(20) - 1.0

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), cn_r.rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["x"]) / z["x"].rolling(60, min_periods=30).var()
        return beta * cn_mom.reindex(s.index)
    return per_asset(f)(close).shift(5)


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


def ic_stats(ic, direction):
    d = 1.0 if ic.mean() >= 0 else -1.0
    direction = direction if direction is not None else d
    adj = direction * ic
    return {"direction": direction, "ic": float(adj.mean()),
            "icir": float(adj.mean() / adj.std()) if len(adj) > 2 and adj.std() > 0 else float("nan"),
            "hit": float((adj > 0).mean()), "n": len(adj)}


def validate(name, factor, close, libsig, window_end=WARM_END, direction=None):
    sub = factor.loc[:window_end]
    res = {"n_dates": int(sub.shape[0])}
    fwd10 = fwd_returns(close, 10)
    ic10 = rank_ic_series(sub, fwd10)
    st = ic_stats(ic10, direction)
    res["ic_h10"] = st["ic"]
    res["icir_h10"] = st["icir"]
    res["direction"] = st["direction"]
    res["hit_h10"] = st["hit"]
    res["n_h10"] = st["n"]
    res["decay"] = {}
    for h in HORIZONS:
        fwd = fwd_returns(close, h)
        ic = rank_ic_series(sub, fwd)
        s = ic_stats(ic, st["direction"])
        res["decay"][str(h)] = round(s["ic"], 4)
    # coverage
    valid = sub.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    ge8 = valid.sum(axis=1) >= MIN_ASSETS
    res["coverage_dates_ge8"] = float(ge8.mean())
    res["n_dates_ge8"] = int(ge8.sum())
    res["turnover_10d_rank"] = turnover_10d_rank(sub)
    res["library_corrs"] = stacked_corr(sub, libsig)
    abs_c = [abs(v) for v in res["library_corrs"].values() if np.isfinite(v)]
    res["max_abs_library_correlation"] = max(abs_c) if abs_c else float("nan")
    res["ic_by_year"] = {}
    yrs = sorted(set(sub.index.year))
    for y in yrs:
        ic_y = rank_ic_series(sub.loc[str(y)], fwd10)
        s_y = ic_stats(ic_y, st["direction"])
        res["ic_by_year"][str(y)] = {"ic": round(s_y["ic"], 4), "icir": round(s_y["icir"], 3) if np.isfinite(s_y["icir"]) else None, "n": int(s_y["n"])}
    # recent 250d window stability
    if len(sub) >= 250:
        last = sub.iloc[-250:]
        ic_l = rank_ic_series(last, fwd10)
        s_l = ic_stats(ic_l, st["direction"])
        res["ic_last250"] = round(s_l["ic"], 4)
        res["icir_last250"] = round(s_l["icir"], 3) if np.isfinite(s_l["icir"]) else None
    res["PASS"] = bool(abs(res["ic_h10"]) >= IC_GATE and abs(res["icir_h10"]) >= ICIR_GATE
                      and res["max_abs_library_correlation"] < CORR_GATE)
    print("=== %s (warm-up ..%s): IC_h10=%+.4f ICIR_h10=%+.4f hit=%.3f n=%d | cov_asset=%.3f cov_ge8=%.3f n_ge8=%d turn=%.2f | maxcorr=%.3f | %s"
          % (name, window_end, res["ic_h10"], res["icir_h10"], res["hit_h10"], res["n_h10"],
             res["coverage_asset_days"], res["coverage_dates_ge8"], res["n_dates_ge8"],
             res["turnover_10d_rank"], res["max_abs_library_correlation"],
             "PASS" if res["PASS"] else "FAIL"), flush=True)
    return res


def live_drift(name, factor, close):
    live = factor.loc[LIVE_START:]
    if len(live) < 60:
        return None
    fwd10 = fwd_returns(close, 10)
    ic = rank_ic_series(live, fwd10)
    st = ic_stats(ic, None)
    return {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
            "start": str(live.index[0].date()), "end": str(live.index[-1].date())}


if __name__ == "__main__":
    close, open_, high, low, vol, macro = load_data()
    print("panel: %d dates x %d assets; warm-up ..%s; data end %s" %
          (close.shape[0], close.shape[1], WARM_END, close.index[-1].date()), flush=True)
    libsig = library_signals(close, macro)
    print("library factors: %s\n" % list(libsig.keys()), flush=True)

    print("##### A) LIBRARY RE-VALIDATION (warm-up admission + live drift) #####", flush=True)
    lib_results = {}
    for fid, sig in libsig.items():
        r = validate("[LIB] " + fid, sig, close, libsig)
        lib_results[fid] = r
        ld = live_drift(fid, sig, close)
        if ld is None:
            print("  %s: too few live obs, skip drift\n" % fid)
        else:
            print("  %s: LIVE %s..%s h10 IC=%+.4f ICIR=%+.4f hit=%.3f n=%d\n" % (fid, ld["start"], ld["end"], ld["ic"], ld["icir"], ld["hit"], ld["n"]))

    print("##### B) ROUND-23 CANDIDATE SCREENS #####", flush=True)
    cands = {
        "bear_beta_60": lambda: cand_bear_beta_60(close),
        "bear_bull_ratio_60": lambda: cand_bear_bull_ratio_60(close),
        "coskew_60": lambda: cand_coskew_60(close),
        "idio_skew_20_skip5": lambda: cand_idio_skew_20_skip5(close),
        "mom_ewma_20_skip5": lambda: cand_mom_ewma_20_skip5(close),
        "spread_beta_cond_60x20": lambda: cand_spread_beta_cond_60x20(close),
        "calmar_60x20_skip5": lambda: cand_calmar_60x20_skip5(close),
        "updown_vol_ratio_20_skip5": lambda: cand_updown_vol_ratio_20_skip5(close),
        "rev_5d_skip1": lambda: cand_rev_5d_skip1(close),
        "mom_tail_cond_20x60": lambda: cand_mom_tail_cond_20x60(close),
        "leverage_effect_20": lambda: cand_leverage_effect_20(close),
        "beta_trend_20x60": lambda: cand_beta_trend_20x60(close),
        "xau_dd_rel_20_skip5": lambda: cand_xau_dd_rel_20_skip5(close),
        "cn10y_beta_cond_60x20": lambda: cand_cn10y_beta_cond_60x20(close),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print("=== %s: ERROR %s: %s ===\n" % (name, type(e).__name__, e), flush=True)

    print("##### SUMMARY #####", flush=True)
    for name, r in results.items():
        print("%-26s IC=%+.4f ICIR=%+.4f maxcorr=%.3f cov_ge8=%.2f turn=%.2f -> %s" %
              (name, r["ic_h10"], r["icir_h10"], r["max_abs_library_correlation"],
               r["coverage_dates_ge8"], r["turnover_10d_rank"], "PASS" if r["PASS"] else "FAIL"), flush=True)

    with open("scripts/miner_1_20270525_screen_results.json", "w") as f:
        json.dump({"candidates": results, "library": lib_results}, f, indent=1, default=str)
    print("saved -> scripts/miner_1_20270525_screen_results.json", flush=True)
