"""
miner_1 cycle 2028-04-11: screen round 29 (fresh factor families).
Universe: 15 tradable cross-asset instruments (warm-up admission 2020-01-01..2026-07-15;
live drift informational 2026-07-16..2028-04-10).
IC = cross-sectional Spearman rank IC per date (>=8 assets), horizon h=10.
Admission gates (benchmark contract): |IC_h10|>=0.007, |ICIR_h10|>=0.084,
max_abs_library_correlation < 0.5 vs the CURRENT 7-factor ensemble library
(rel_mom_20d_skip5, downside_vol_ratio_20, beta_ew_60d, corr_ew_60, kurt_20d_skip5,
dxy_beta_cond_60x20, max_ret_20d).

Round-29 candidates (families NOT screened in rounds 1-28 / by any miner):
  hurst_60            - R/S Hurst exponent over 60d returns (trend persistence, H>0.5 trending)
  r2_trend_60         - R^2 of OLS log(close)~time over 60d (trend quality / fit strength)
  ar1_hl_60           - AR(1) coefficient of daily returns over 60d (+ momentum persistence, - reversal)
  cokurt_60           - co-kurtosis of asset returns vs EW market returns over 60d (systematic tail)
  capm_alpha_60       - 60d CAPM alpha vs EW market (excess return after beta adjustment)
  down_skew_60        - skewness of returns on EW-down days over 60d (down-market tail shape)
  days_since_low_60   - days elapsed since rolling 60d low (recovery/trend age)
  skew_trend_20x60    - skew60 - skew20 (skewness drift / tail-shape evolution)
  dxy_cond_mom_20x60  - 20d momentum (skip5) x sign(DXY 60d momentum) (USD-regime gated momentum)
  vwap_dev_20         - close vs 20d VWAP deviation (volume-weighted; note vol missing for 6 assets)
"""
from __future__ import annotations
import json
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

LIB_FACTORS = ["rel_mom_20d_skip5", "downside_vol_ratio_20", "beta_ew_60d",
               "corr_ew_60", "kurt_20d_skip5", "dxy_beta_cond_60x20", "max_ret_20d"]


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
    idx = factor.index.intersection(fwd.index)
    F = factor.loc[idx].rank(axis=1).values.astype(float)
    R = fwd.loc[idx].rank(axis=1).values.astype(float)
    return pd.Series(row_pearson(F, R), index=idx)


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


# ---------------- current effective library (7 factors, ensemble definitions) ----------------
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

    # corr_ew_60: mean pairwise rolling 60d correlation with every other asset
    pairs = {}
    assets = list(close.columns)
    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            z = pd.concat([close[a].pct_change().rename("x"), close[b].pct_change().rename("y")], axis=1).dropna()
            pairs[(a, b)] = z["x"].rolling(60, min_periods=30).corr(z["y"])
    cols = {}
    for a in assets:
        df = pd.concat([v for (x, y), v in pairs.items() if x == a or y == a], axis=1)
        cols[a] = df.mean(axis=1)
    lib["corr_ew_60"] = pd.DataFrame(cols, index=close.index)

    def kurt(s):
        rr = s.pct_change()
        return rr.rolling(20, min_periods=10).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close).shift(5)

    dxy = macro["DXY"]
    dxy_r = dxy.pct_change()
    dxy_mom = dxy / dxy.shift(20) - 1.0

    def dxy_beta_cond(s):
        z = pd.concat([s.pct_change().rename("r"), dxy_r.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["x"]) / z["x"].rolling(60, min_periods=30).var()
        return -beta * dxy_mom.reindex(s.index)
    lib["dxy_beta_cond_60x20"] = per_asset(dxy_beta_cond)(close)

    lib["max_ret_20d"] = r.rolling(20).max()
    return lib


# ---------------- round-29 candidates ----------------
def _ew_ret(close):
    return close.pct_change().mean(axis=1)


def cand_hurst_60(close):
    """R/S Hurst exponent, 60d window of daily log returns, skip5. H = log(R/S)/log(n)."""
    def f(s):
        lr = np.log(s).diff()
        out = {}
        idx = s.index
        for t in range(len(s)):
            if t < 64 or not np.isfinite(lr.iloc[t - 5]):
                out[idx[t]] = np.nan
                continue
            w = lr.iloc[t - 5 - 59:t - 5].values
            w = w[np.isfinite(w)]
            if len(w) < 40 or w.std() < EPS:
                out[idx[t]] = np.nan
                continue
            c = np.cumsum(w - w.mean())
            rs = (c.max() - c.min()) / w.std()
            out[idx[t]] = float(np.log(rs + EPS) / np.log(len(w)))
        return pd.Series(out)
    return per_asset(f)(close)


def cand_r2_trend_60(close):
    """R^2 of OLS log(close) ~ time over 60d, skip5."""
    def f(s):
        lx = np.log(s)
        out = {}
        idx = s.index
        tarr = np.arange(60, dtype=float)
        for t in range(len(s)):
            if t < 64 or not np.isfinite(lx.iloc[t - 5]):
                out[idx[t]] = np.nan
                continue
            y = lx.iloc[t - 5 - 59:t - 5].values
            if not np.all(np.isfinite(y)):
                out[idx[t]] = np.nan
                continue
            b = np.polyfit(tarr, y, 1)
            yhat = np.polyval(b, tarr)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            out[idx[t]] = float(1.0 - ss_res / (ss_tot + EPS))
        return pd.Series(out)
    return per_asset(f)(close)


def cand_ar1_hl_60(close):
    """AR(1) coefficient of daily returns over 60d, skip5."""
    def f(s):
        rr = s.pct_change()
        z = pd.concat([rr.shift(1).rename("lag"), rr.rename("r")], axis=1).dropna()
        return z["lag"].rolling(60, min_periods=30).cov(z["r"]) / z["lag"].rolling(60, min_periods=30).var()
    return per_asset(f)(close).shift(5)


def cand_cokurt_60(close):
    """Co-kurtosis vs EW market: E[(r_i)(m)^3] / (std_i * std_m^3), 60d demeaned, skip5."""
    ew_r = _ew_ret(close)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        ri = z["r"] - z["r"].rolling(60, min_periods=30).mean()
        mi = z["m"] - z["m"].rolling(60, min_periods=30).mean()
        num = (ri * mi ** 3).rolling(60, min_periods=30).mean()
        den = z["r"].rolling(60, min_periods=30).std() * z["m"].rolling(60, min_periods=30).std() ** 3
        return num / (den + EPS)
    return per_asset(f)(close).shift(5)


def cand_capm_alpha_60(close):
    """60d CAPM alpha vs EW: mean(r) - beta*mean(m), skip5."""
    ew_r = _ew_ret(close)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60, min_periods=30).cov(z["m"]) / z["m"].rolling(60, min_periods=30).var()
        alpha = z["r"].rolling(60, min_periods=30).mean() - beta * z["m"].rolling(60, min_periods=30).mean()
        return alpha
    return per_asset(f)(close).shift(5)


def cand_down_skew_60(close):
    """Skewness of asset returns on EW-down days over 60d, skip5."""
    ew_r = _ew_ret(close)

    def f(s):
        rr = s.pct_change()
        mask = (ew_r < 0).astype(float)
        mask = mask.where(mask == 1, np.nan)
        return (rr * mask).rolling(60, min_periods=20).skew()
    return per_asset(f)(close).shift(5)


def cand_days_since_low_60(close):
    """Days elapsed since the rolling 60d minimum (0 = at low), skip5."""
    def f(s):
        lows = s.rolling(60, min_periods=30).min()
        cnt = pd.Series(0, index=s.index)
        days = pd.Series(np.nan, index=s.index)
        cur = 0
        low_val = np.inf
        for t in range(len(s)):
            v = s.iloc[t]
            if not np.isfinite(v):
                days.iloc[t] = np.nan
                continue
            cur += 1
            if np.isfinite(lows.iloc[t]) and v <= lows.iloc[t] + EPS:
                cur = 0
                low_val = v
            days.iloc[t] = cur
        return days
    return per_asset(f)(close).shift(5)


def cand_skew_trend_20x60(close):
    """skew60 - skew20 (both skip5): skewness drift."""
    def sk(s, w):
        return s.pct_change().rolling(w, min_periods=max(10, w // 2)).skew()
    sk60 = per_asset(lambda s: sk(s, 60))(close).shift(5)
    sk20 = per_asset(lambda s: sk(s, 20))(close).shift(5)
    return sk60 - sk20


def cand_dxy_cond_mom_20x60(close, macro):
    """20d momentum (skip5) x sign(DXY 60d momentum)."""
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    dxy = macro["DXY"]
    dxy_mom60 = np.sign(dxy / dxy.shift(60) - 1.0)
    return m20.mul(dxy_mom60, axis=0)


def cand_vwap_dev_20(close, high, low, vol):
    """(close - 20d VWAP)/VWAP; VWAP = sum(typical*vol)/sum(vol); volume-gated."""
    typical = (high + low + close) / 3.0
    pv = (typical * vol).rolling(20, min_periods=10).sum()
    v = vol.rolling(20, min_periods=10).sum()
    vwap = pv / (v + EPS)
    return (close / vwap - 1.0).shift(5)


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
    fwd10 = fwd_returns(close, 10).reindex(sub.index)
    ic10 = rank_ic_series(sub, fwd10)
    st = ic_stats(ic10, direction)
    res["ic_h10"] = st["ic"]
    res["icir_h10"] = st["icir"]
    res["direction"] = st["direction"]
    res["hit_h10"] = st["hit"]
    res["n_h10"] = st["n"]
    res["decay"] = {}
    for h in HORIZONS:
        fwd = fwd_returns(close, h).reindex(sub.index)
        ic = rank_ic_series(sub, fwd)
        s = ic_stats(ic, st["direction"])
        res["decay"][str(h)] = round(s["ic"], 4)
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
        res["ic_by_year"][str(y)] = {"ic": round(s_y["ic"], 4),
                                     "icir": round(s_y["icir"], 3) if np.isfinite(s_y["icir"]) else None,
                                     "n": int(s_y["n"])}
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
    fwd10 = fwd_returns(close, 10).reindex(live.index)
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
            print("  %s: LIVE %s..%s h10 IC=%+.4f ICIR=%+.4f hit=%.3f n=%d\n" %
                  (fid, ld["start"], ld["end"], ld["ic"], ld["icir"], ld["hit"], ld["n"]))

    print("##### B) ROUND-29 CANDIDATE SCREENS #####", flush=True)
    cands = {
        "hurst_60": lambda: cand_hurst_60(close),
        "r2_trend_60": lambda: cand_r2_trend_60(close),
        "ar1_hl_60": lambda: cand_ar1_hl_60(close),
        "cokurt_60": lambda: cand_cokurt_60(close),
        "capm_alpha_60": lambda: cand_capm_alpha_60(close),
        "down_skew_60": lambda: cand_down_skew_60(close),
        "days_since_low_60": lambda: cand_days_since_low_60(close),
        "skew_trend_20x60": lambda: cand_skew_trend_20x60(close),
        "dxy_cond_mom_20x60": lambda: cand_dxy_cond_mom_20x60(close, macro),
        "vwap_dev_20": lambda: cand_vwap_dev_20(close, high, low, vol),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
            ld = live_drift(name, factor, close)
            if ld is not None:
                print("  LIVE %s..%s h10 IC=%+.4f ICIR=%+.4f hit=%.3f n=%d\n" %
                      (ld["start"], ld["end"], ld["ic"], ld["icir"], ld["hit"], ld["n"]))
        except Exception as e:
            print("=== %s: ERROR %s: %s ===\n" % (name, type(e).__name__, e), flush=True)

    print("##### SUMMARY #####", flush=True)
    for name, r in results.items():
        print("%-22s IC=%+.4f ICIR=%+.4f maxcorr=%.3f cov_ge8=%.2f turn=%.2f -> %s" %
              (name, r["ic_h10"], r["icir_h10"], r["max_abs_library_correlation"],
               r["coverage_dates_ge8"], r["turnover_10d_rank"], "PASS" if r["PASS"] else "FAIL"), flush=True)

    with open("scripts/miner_1_20280411_screen_results.json", "w") as f:
        json.dump({"candidates": results, "library": lib_results}, f, indent=1, default=str)
    print("saved -> scripts/miner_1_20280411_screen_results.json", flush=True)
