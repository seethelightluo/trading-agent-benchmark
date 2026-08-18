"""
miner_1 cycle 2028-06-06: screen round 31.
Universe: 15 tradable cross-asset instruments (warm-up admission 2020-01-01..2026-07-15;
live drift informational 2026-07-16..data end).
IC = cross-sectional Spearman rank IC per date (>=8 assets), admission horizon h=10.
Admission gates (benchmark contract): |IC_h10|>=0.007, |ICIR_h10|>=0.084.
max_abs_library_correlation reported as audit metadata (deterministic post-Miner gate
recomputes rho from real signal artifacts).

Context: 2028-06-06 trader cycle - SIDEWAYS-to-BEAR, VIX 30.25 HIGH, risk-off def_floor,
frozen feeds NDX/SOX/000688/CN10Y, USD strength (DXY 101.13 +3.0%/60d), yields RISING
(US10Y bond-floor hazard), rel_mom whipsaw on crypto/commodity flagged by screener.

Round-31 candidates:
  REVALIDATE (round-29 PASS, never persisted):
    ar1_hl_60           - AR(1) coeff of daily returns over 60d, skip5 (persistence/reversal)
    down_skew_60        - skewness of returns on EW-down days over 60d, skip5 (tail shape)
    skew_trend_20x60    - skew60 - skew20, skip5 (skewness drift)
    vwap_dev_20         - (close - 20d VWAP)/VWAP, skip5 (volume-price deviation)
  NEW round-31:
    price_pos_20        - (close - min20)/(max20 - min20), skip5 (range position)
    sharpe_20x60        - 20d mom (skip5) / 20d vol (risk-adjusted momentum)
    vol_ratio_10x60     - 10d vol / 60d vol, skip5 (short-term vol regime expansion)
    trend_gated_mom_20x60 - mom20_skip5 * (close > sma60) (trend-confirmed momentum,
                            addresses rel_mom whipsaw flag)
    us10y_beta_cond_60x20 - beta to US10Y * US10Y 20d mom (duration regime conditional)
    xau_beta_cond_60x20   - beta to XAU * XAU 20d mom (defensive rotation conditional)
    zscore_20           - (close - sma20)/std20, skip5 (deviation z-score)
    ret_60_skip20       - 60d return skipping last 20d (longer-horizon trend)
"""
from __future__ import annotations
import json
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
IC_GATE, ICIR_GATE = 0.007, 0.084
HORIZONS = (1, 2, 3, 5, 10, 20)
EPS = 1e-12


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


def stacked_corr(cand, libsig):
    out = {}
    f = cand.stack().rename("f")
    for fid, ls in libsig.items():
        g = ls.stack().rename("g")
        j = pd.concat([f, g], axis=1).dropna()
        if len(j) < 100:
            out[fid] = float("nan")
            continue
        out[fid] = float(j["f"].corr(j["g"], method="spearman"))
    return out


# ---------------- library (7 effective ensemble factors) ----------------
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
def cand_ar1_hl_60(close):
    def f(s):
        rr = s.pct_change()
        z = pd.concat([rr.shift(1).rename("lag"), rr.rename("r")], axis=1).dropna()
        return z["lag"].rolling(60, min_periods=30).cov(z["r"]) / z["lag"].rolling(60, min_periods=30).var()
    return per_asset(f)(close).shift(5)


def cand_down_skew_60(close):
    ew_r = close.mean(axis=1).pct_change()

    def f(s):
        rr = s.pct_change()
        mask = (ew_r < 0).astype(float)
        mask = mask.where(mask == 1, np.nan)
        return (rr * mask).rolling(60, min_periods=20).skew()
    return per_asset(f)(close).shift(5)


def cand_skew_trend_20x60(close):
    def sk(s, w):
        return s.pct_change().rolling(w, min_periods=max(10, w // 2)).skew()
    sk60 = per_asset(lambda s: sk(s, 60))(close).shift(5)
    sk20 = per_asset(lambda s: sk(s, 20))(close).shift(5)
    return sk60 - sk20


def cand_vwap_dev_20(close, high, low, vol):
    typical = (high + low + close) / 3.0
    pv = (typical * vol).rolling(20, min_periods=10).sum()
    v = vol.rolling(20, min_periods=10).sum()
    vwap = pv / (v + EPS)
    return (close / vwap - 1.0).shift(5)


def cand_price_pos_20(close):
    def f(s):
        mn = s.rolling(20, min_periods=10).min()
        mx = s.rolling(20, min_periods=10).max()
        return (s - mn) / (mx - mn + EPS)
    return per_asset(f)(close).shift(5)


def cand_sharpe_20x60(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    v20 = per_asset(lambda s: s.pct_change().rolling(20, min_periods=10).std())(close)
    return m20 / (v20 + EPS)


def cand_vol_ratio_10x60(close):
    v10 = per_asset(lambda s: s.pct_change().rolling(10, min_periods=6).std())(close)
    v60 = per_asset(lambda s: s.pct_change().rolling(60, min_periods=30).std())(close)
    return (v10 / (v60 + EPS)).shift(5)


def cand_trend_gated_mom_20x60(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    above = close > close.rolling(60, min_periods=30).mean()
    return m20 * above.astype(float)


def cand_ref_beta_cond(close, ref, window=60, mom=20):
    ref20 = (ref / ref.shift(mom) - 1.0)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(window).cov(z["x"]) / z["x"].rolling(window).var()
        return beta * ref20.reindex(s.index)
    return per_asset(f)(close)


def cand_zscore_20(close):
    def f(s):
        mu = s.rolling(20, min_periods=10).mean()
        sd = s.rolling(20, min_periods=10).std()
        return (s - mu) / (sd + EPS)
    return per_asset(f)(close).shift(5)


def cand_ret_60_skip20(close):
    return per_asset(lambda s: s.shift(20) / s.shift(80) - 1.0)(close)


# ---------------- validation ----------------
def validate(name, factor, close, libsig, direction=1.0):
    res = {"name": name, "n_dates": int(factor.shape[0])}
    window_end = pd.Timestamp(WARM_END)
    ic10 = rank_ic_series(factor.loc[:window_end], fwd_returns(close, 10))
    res["n_h10"] = len(ic10)
    res["ic_h10"] = float(direction * ic10.mean()) if len(ic10) else float("nan")
    res["icir_h10"] = float(direction * ic10.mean() / ic10.std()) if len(ic10) > 2 and ic10.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic10 > 0).mean()) if len(ic10) else float("nan")
    res["decay"] = {}
    for h in HORIZONS:
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
    res["PASS"] = bool(gate)

    # live + recent12m drift (informational)
    if factor.index[-1] > pd.Timestamp(LIVE_START):
        for tag, sl in (("live", slice(LIVE_START, None)),
                        ("recent12m", slice(pd.Timestamp("2027-06-06"), None))):
            try:
                ic_l = rank_ic_series(factor.loc[sl], fwd_returns(close, 10))
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
        print(f"  LIVE: IC={res['live_ic_h10']:+.4f} ICIR={res['live_icir_h10']:+.4f} n={res['live_n']} | "
              f"recent12m IC={res['recent12m_ic_h10']:+.4f} ICIR={res['recent12m_icir_h10']:+.4f} n={res['recent12m_n']}")
    print(f"  gate: |IC|>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | "
          f"|ICIR|>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


if __name__ == "__main__":
    close, open_, high, low, vol, macro = load_data()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}")
    print(f"library factors: {list(libsig.keys())}\n")
    print("regime sanity: ", end="")
    for s in ["VIX", "DXY", "EURUSD", "USDJPY", "USDCNY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%} ", end="")
    print("\n")

    cands = {
        "ar1_hl_60": (cand_ar1_hl_60(close), -1.0),
        "down_skew_60": (cand_down_skew_60(close), -1.0),
        "skew_trend_20x60": (cand_skew_trend_20x60(close), -1.0),
        "vwap_dev_20": (cand_vwap_dev_20(close, high, low, vol), 1.0),
        "price_pos_20": (cand_price_pos_20(close), 1.0),
        "sharpe_20x60": (cand_sharpe_20x60(close), 1.0),
        "vol_ratio_10x60": (cand_vol_ratio_10x60(close), -1.0),
        "trend_gated_mom_20x60": (cand_trend_gated_mom_20x60(close), 1.0),
        "us10y_beta_cond_60x20": (cand_ref_beta_cond(close, macro["US10Y_ref"] if "US10Y_ref" in macro else macro.get("US10Y", pd.Series(dtype=float)), 60, 20), 1.0),
        "xau_beta_cond_60x20": (cand_ref_beta_cond(close, close["XAU"].dropna(), 60, 20), 1.0),
        "zscore_20": (cand_zscore_20(close), 1.0),
        "ret_60_skip20": (cand_ret_60_skip20(close), 1.0),
    }
    # fix us10y ref: use US10Y asset close series itself
    cands["us10y_beta_cond_60x20"] = (cand_ref_beta_cond(close, close["US10Y"].dropna(), 60, 20), 1.0)

    results = {}
    for name, (fac, direction) in cands.items():
        try:
            results[name] = validate(name, fac, close, libsig, direction=direction)
        except Exception as e:
            print(f"=== {name} === ERROR {e}\n")
            results[name] = {"name": name, "PASS": False, "error": str(e)}

    with open("scripts/miner_1_20280606_screen_results.json", "w") as fh:
        json.dump({"date": "2028-06-06", "results": results}, fh, indent=1, default=str)

    print("\n##### SUMMARY #####")
    for k, v in results.items():
        print(f"{k}: IC={v.get('ic_h10', float('nan')):+.4f} ICIR={v.get('icir_h10', float('nan')):+.4f} "
              f"maxcorr={v.get('max_abs_library_correlation', float('nan')):.3f} PASS={v.get('PASS')}")
