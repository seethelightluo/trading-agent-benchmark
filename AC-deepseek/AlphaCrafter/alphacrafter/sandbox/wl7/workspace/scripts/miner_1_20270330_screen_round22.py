"""
miner_1 cycle 2027-03-30: screen round 22 (fresh factor families).
Universe: 15 tradable cross-asset instruments (2020-01-01..2026-07-15 warm-up
admission; live drift informational 2026-07-16..2027-03-29).
IC = cross-sectional Spearman rank IC per date (>=8 assets).
Admission gates (benchmark contract): |IC_h10|>=0.007, |ICIR_h10|>=0.084,
max_abs_library_correlation < 0.5 vs the CURRENT 7-factor effective library.

Round-22 candidates (fresh families not screened in rounds 1-21):
  overnight_gap_corr_20 - 20d corr(overnight gap ret, intraday ret) skip5
                          (overnight-vs-intraday return asymmetry)
  mom_accel_20x40       - 20d momentum minus 40d momentum, skip5 (acceleration)
  xau_rel_mom_20        - 20d momentum relative to XAU, skip5 (safe-haven anchor)
  upvol_share_20        - up-day volume share over 20d, skip5 (volume breadth)
  rank_mom_5            - 5d change in cross-sectional rank of 20d momentum
  dd_vol_ratio_60       - 60d drawdown depth / 60d realized vol
  kelly_20_skip5        - mean/std of daily returns over 20d, skip5 (short Sharpe)
  idio_vol_20           - 20d idiosyncratic vol vs EW-market regression, skip5
  rsi_2_skip2           - Wilder RSI(2), skip2 (short-term oscillator)
  di_diff_14            - Wilder +DI(14) - -DI(14), skip2 (trend direction)
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


# ---------------- round-22 candidates ----------------
def cand_overnight_gap_corr_20(close, open_):
    prev_close = close.shift(1)
    gap = open_ / prev_close - 1.0
    intra = close / open_ - 1.0
    c = gap.rolling(20, min_periods=10).corr(intra)
    return c.shift(5)


def cand_mom_accel_20x40(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    m40 = per_asset(lambda s: s.shift(5) / s.shift(45) - 1.0)(close)
    return m20 - m40


def cand_xau_rel_mom_20(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    xau_mom = m20["XAU"]
    return m20.sub(xau_mom, axis=0)


def cand_upvol_share_20(close, vol):
    r = close.pct_change()
    up = (r > 0).astype(float)
    vup = vol * up
    share = vup.rolling(20, min_periods=10).sum() / vol.rolling(20, min_periods=10).sum()
    return share.shift(5)


def cand_rank_mom_5(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    rk = m20.rank(axis=1)
    return rk - rk.shift(5)


def cand_dd_vol_ratio_60(close):
    dd = close / close.rolling(60, min_periods=30).max() - 1.0
    v = close.pct_change().rolling(60, min_periods=30).std()
    return (dd / (v + EPS)).shift(5)


def cand_kelly_20_skip5(close):
    r = close.pct_change()
    mu = r.rolling(20, min_periods=10).mean()
    sd = r.rolling(20, min_periods=10).std()
    return (mu / (sd + EPS)).shift(5)


def cand_idio_vol_20(close):
    ew_r = close.pct_change().mean(axis=1)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
        resid = z["r"] - beta * z["m"]
        return resid.rolling(20, min_periods=10).std()
    return per_asset(f)(close).shift(5)


def cand_rsi_2_skip2(close):
    def rsi2(s):
        delta = s.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        ag = gain.ewm(alpha=0.5, min_periods=2).mean()
        al = loss.ewm(alpha=0.5, min_periods=2).mean()
        rs = ag / (al + EPS)
        return 100.0 - 100.0 / (1.0 + rs)
    return per_asset(rsi2)(close).shift(2)


def cand_di_diff_14(close, high, low):
    def f(s, h, l):
        up = h.diff()
        dn = -l.diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=s.index)
        mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=s.index)
        tr = pd.concat([h - l, (h - s.shift()).abs(), (l - s.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 14, min_periods=14).mean()
        pdi = 100.0 * pdm.ewm(alpha=1.0 / 14, min_periods=14).mean() / (atr + EPS)
        mdi = 100.0 * mdm.ewm(alpha=1.0 / 14, min_periods=14).mean() / (atr + EPS)
        return pdi - mdi
    cols = {}
    for a in close.columns:
        s = close[a].dropna()
        cols[a] = f(s, high[a].reindex(s.index), low[a].reindex(s.index))
    return pd.DataFrame(cols, index=close.index).shift(2)


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
        ic_h = rank_ic_series(sub, fwd_returns(close, h))
        res["decay"][str(h)] = round(st["direction"] * ic_h.mean(), 4) if len(ic_h) else float("nan")
    valid = sub.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(sub)
    corrs = stacked_corr(sub, libsig)
    res["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))}
    res["ic_by_year"] = {}
    for yr, grp in ic10.groupby(ic10.index.year):
        ys = ic_stats(grp, st["direction"])
        res["ic_by_year"][str(yr)] = {"ic": round(ys["ic"], 4), "icir": round(ys["icir"], 4), "n": int(ys["n"])}
    gate = abs(res["ic_h10"]) >= IC_GATE and abs(res["icir_h10"]) >= ICIR_GATE
    lowcorr = res["max_abs_library_correlation"] < CORR_GATE
    res["PASS"] = bool(gate and lowcorr)
    print("=== %s === dates=%d direction=%+.2f" % (name, res["n_dates"], st["direction"]))
    print("  h10 IC=%+.4f ICIR=%+.4f hit=%.3f n=%d" % (res["ic_h10"], res["icir_h10"], res["hit_h10"], res["n_h10"]))
    print("  decay=%s" % res["decay"])
    print("  cov_asset=%.3f cov_ge8=%.3f turn=%.3f" % (res["coverage_asset_days"], res["coverage_dates_ge8"], res["turnover_10d_rank"]))
    print("  by_year=%s" % res["ic_by_year"])
    print("  max_lib_corr=%.3f corrs=%s" % (res["max_abs_library_correlation"], res["library_corrs"]))
    ok_ic = abs(res["ic_h10"]) >= IC_GATE
    ok_icir = abs(res["icir_h10"]) >= ICIR_GATE
    print("  gate: IC>=" + str(IC_GATE) + (" OK" if ok_ic else " FAIL") +
          " | ICIR>=" + str(ICIR_GATE) + (" OK" if ok_icir else " FAIL") +
          " | corr<" + str(CORR_GATE) + (" OK" if lowcorr else " FAIL") +
          " -> " + ("PASS" if res["PASS"] else "FAIL") + "\n")
    return res


def live_drift(fid, sig, close):
    sub = sig.loc[LIVE_START:]
    if sub.notna().sum().sum() < 200:
        return None
    ic = rank_ic_series(sub, fwd_returns(close, 10).loc[LIVE_START:])
    if len(ic) < 5:
        return None
    d = 1.0 if ic.mean() >= 0 else -1.0
    icir = d * ic.mean() / ic.std() if ic.std() > 0 else float("nan")
    return {"ic": round(d * ic.mean(), 4), "icir": round(icir, 4),
            "hit": round(float((d * ic > 0).mean()), 3), "n": int(len(ic))}


# ---------------- persistence ----------------
def save_artifact_csv(factor):
    df = factor.copy()
    df.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    buf = io.StringIO()
    df.to_csv(buf)
    return base64.b64encode(zlib.compress(buf.getvalue().encode())).decode()


def persist_factor(name, fname, expr, desc, deps, params, res, tags, regime, close, factor):
    direction = res["direction"]
    factor_full = factor.reindex(close.index)
    npy_path = Path("factors") / (name + ".signal.npy")
    artifact = np.asarray(factor_full.values, dtype=float)
    np.save(npy_path, artifact)
    payload = {
        "factor_id": name, "factor_name": fname, "version": "1.0.0",
        "calculation": {"expression": expr, "description": desc},
        "dependencies": deps, "parameters": params,
        "expected_direction": int(np.sign(direction)) if direction != 0 else 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01.." + WARM_END,
            "last_validated": "2027-03-30",
            "admission_horizon": 10,
            "regime_notes": regime,
            "metrics": {
                "ic": res["ic_h10"], "icir": res["icir_h10"],
                "ic_hit_ratio": res["hit_h10"], "n_ic_dates": res["n_h10"],
                "coverage_asset_days": res["coverage_asset_days"],
                "coverage_dates_ge8": res["coverage_dates_ge8"],
                "turnover_10d_rank": res["turnover_10d_rank"],
                "decay_ic_by_horizon": res["decay"],
                "max_abs_library_correlation": res["max_abs_library_correlation"],
                "library_pairwise_corr": res["library_corrs"],
                "ic_by_year": res["ic_by_year"],
            },
            "signal_artifact": {
                "format": "base64:zlib:csv",
                "description": "Factor signal panel 15 assets x dates",
                "csv_base64_zlib": save_artifact_csv(factor_full),
            },
        },
        "tags": tags,
    }
    with open(Path("factors") / (name + ".json"), "w") as f:
        json.dump(payload, f, indent=1)
    print("persisted -> factors/%s.json (+ .signal.npy)" % name, flush=True)


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
            print("  %s: LIVE h10 IC=%+.4f ICIR=%+.4f hit=%.3f n=%d\n" % (fid, ld["ic"], ld["icir"], ld["hit"], ld["n"]))

    print("##### B) ROUND-22 CANDIDATE SCREENS #####", flush=True)
    cands = {
        "overnight_gap_corr_20": lambda: cand_overnight_gap_corr_20(close, open_),
        "mom_accel_20x40": lambda: cand_mom_accel_20x40(close),
        "xau_rel_mom_20": lambda: cand_xau_rel_mom_20(close),
        "upvol_share_20": lambda: cand_upvol_share_20(close, vol),
        "rank_mom_5": lambda: cand_rank_mom_5(close),
        "dd_vol_ratio_60": lambda: cand_dd_vol_ratio_60(close),
        "kelly_20_skip5": lambda: cand_kelly_20_skip5(close),
        "idio_vol_20": lambda: cand_idio_vol_20(close),
        "rsi_2_skip2": lambda: cand_rsi_2_skip2(close),
        "di_diff_14": lambda: cand_di_diff_14(close, high, low),
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
        print("%-24s IC=%+.4f ICIR=%+.4f maxcorr=%.3f cov_ge8=%.2f turn=%.2f -> %s" %
              (name, r["ic_h10"], r["icir_h10"], r["max_abs_library_correlation"],
               r["coverage_dates_ge8"], r["turnover_10d_rank"], "PASS" if r["PASS"] else "FAIL"), flush=True)

    with open("scripts/miner_1_20270330_screen_results.json", "w") as f:
        json.dump({"candidates": results, "library": lib_results}, f, indent=1, default=str)
    print("saved -> scripts/miner_1_20270330_screen_results.json", flush=True)
