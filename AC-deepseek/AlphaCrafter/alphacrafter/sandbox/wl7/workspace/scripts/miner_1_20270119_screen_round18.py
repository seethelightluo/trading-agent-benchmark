"""
miner_1 cycle 2027-01-19: screen round 18 (fresh factor families).
Universe: 15 tradable cross-asset instruments (2020-01-01..2026-07-15 warm-up
admission; live drift informational 2026-07-16..2027-01-18).
IC = cross-sectional Spearman rank IC per date (>=8 assets).
Admission gates (benchmark contract): |IC_h10|>=0.007, |ICIR_h10|>=0.084,
max_abs_library_correlation < 0.5 vs the CURRENT 7-factor effective library.

Round-18 candidates (not previously screened in rounds 1-17):
  rsi_14_skip2          - Wilder RSI(14) with 2d skip (oscillator/mean-reversion)
  win_rate_20d_skip5    - share of positive daily returns over 20d, skip 5 (trend breadth)
  updown_capture_60     - mean up-day return / |mean down-day return| over 60d (asymmetry)
  zscore_60d            - (close - sma60)/std60 (trend/mean-reversion z-score)
  hl_range_pos_20       - (close - min20)/(max20 - min20) (stochastic position)
  corr_ew_drift_60x20   - 20d change in corr vs EW market (comovement drift)
  skew_20d_skip5        - 20d return skewness, skip 5 (tail asymmetry)
  dd_depth_60           - close/max60 - 1 (drawdown depth)
  vol_zscore_20x120     - z-score of 20d vol vs its own trailing 120d distribution
  ewma_cross_20x60      - ewma(20)/ewma(60) - 1 (MA crossover trend)
  rates_beta_cond_60x20 - beta(asset, US10Y ret, 60d) * US10Y 20d momentum
                          (conditional yield-beta; rising-yield regime hedge)
  crypto_beta_cond_20   - beta(asset, BTC ret, 60d) * BTC 20d momentum
  overnight_share_20    - |open-prev_close| share of total daily move (overnight vs intraday)
  rel_vol_20            - 20d vol / cross-sectional median 20d vol
  mom60_skip5_rel       - 60d relative momentum, skip 5 (longer-horizon trend)
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
            print(f"WARN no data {s}")
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

    def fx_cond(ref):
        ref20 = (ref / ref.shift(20) - 1.0)

        def f(s):
            z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
            return beta * ref20.reindex(s.index)
        return per_asset(f)(close)
    lib["eurusd_beta_cond_60x20"] = fx_cond(macro["EURUSD"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-18 candidates ----------------
def cand_rsi_14_skip2(close):
    r = close.pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rs = up / (dn + EPS)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.shift(2)


def cand_win_rate_20d_skip5(close):
    r = close.pct_change().shift(5)
    return (r > 0).rolling(20, min_periods=12).mean()


def cand_updown_capture_60(close):
    r = close.pct_change()
    up = r.clip(lower=0).rolling(60, min_periods=30).mean()
    dn = (-r.clip(upper=0)).rolling(60, min_periods=30).mean()
    return up / (dn + EPS)


def cand_zscore_60d(close):
    sma = close.rolling(60, min_periods=30).mean()
    sd = close.rolling(60, min_periods=30).std()
    return (close - sma) / (sd + EPS)


def cand_hl_range_pos_20(close):
    hi = close.rolling(20, min_periods=10).max()
    lo = close.rolling(20, min_periods=10).min()
    return (close - lo) / (hi - lo + EPS)


def cand_corr_ew_drift_60x20(close):
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()

    def ew_corr(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).corr(z["m"])
    c60 = per_asset(ew_corr)(close)
    return c60 - c60.shift(20)


def cand_skew_20d_skip5(close):
    r = close.pct_change().shift(5)

    def f(s):
        return s.rolling(20, min_periods=12).skew()
    return per_asset(f)(r)


def cand_dd_depth_60(close):
    return close / close.rolling(60, min_periods=30).max() - 1.0


def cand_vol_zscore_20x120(close):
    r = close.pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    mu = v20.rolling(120, min_periods=60).mean()
    sd = v20.rolling(120, min_periods=60).std()
    return (v20 - mu) / (sd + EPS)


def cand_ewma_cross_20x60(close):
    e20 = close.ewm(span=20, adjust=False).mean()
    e60 = close.ewm(span=60, adjust=False).mean()
    return e20 / (e60 + EPS) - 1.0


def cand_rates_beta_cond_60x20(close, macro):
    u10 = macro["US10Y"].dropna()
    u10_mom = (u10 / u10.shift(20) - 1.0)
    r10 = u10.pct_change()

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), r10.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        return beta * u10_mom.reindex(s.index)
    return per_asset(f)(close)


def cand_crypto_beta_cond_20(close):
    btc = close["BTC"].pct_change()
    btc_mom = (close["BTC"] / close["BTC"].shift(20) - 1.0)

    def f(s):
        z = pd.concat([s.pct_change().rename("r"), btc.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        return beta * btc_mom.reindex(s.index)
    return per_asset(f)(close)


def cand_overnight_share_20(close, open_):
    prev_close = close.shift(1)
    gap = (open_ - prev_close).abs()
    intra = (close - open_).abs()
    return gap.rolling(20, min_periods=10).mean() / (gap + intra).rolling(20, min_periods=10).mean()


def cand_rel_vol_20(close):
    r = close.pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    return v20.div(v20.median(axis=1), axis=0)


def cand_mom60_skip5_rel(close):
    m60 = per_asset(lambda s: s.shift(5) / s.shift(65) - 1.0)(close)
    return m60.sub(m60.median(axis=1), axis=0)


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
    res.update({f"ic_h{k}": st["ic"], f"icir_h{k}": st["icir"] for k in [10]})
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
    # per-year
    res["ic_by_year"] = {}
    for yr, grp in ic10.groupby(ic10.index.year):
        ys = ic_stats(grp, st["direction"])
        res["ic_by_year"][str(yr)] = {"ic": round(ys["ic"], 4), "icir": round(ys["icir"], 4), "n": int(ys["n"])}
    gate = abs(res["ic_h10"]) >= IC_GATE and abs(res["icir_h10"]) >= ICIR_GATE
    lowcorr = res["max_abs_library_correlation"] < CORR_GATE
    res["PASS"] = bool(gate and lowcorr)
    print(f"=== {name} === dates={res['n_dates']} direction={st['direction']:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  decay={res['decay']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  by_year={res['ic_by_year']}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} "
          f"{'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


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
    npy_path = Path("factors") / f"{name}.signal.npy"
    artifact = np.asarray(factor_full.values, dtype=float)
    np.save(npy_path, artifact)
    payload = {
        "factor_id": name, "factor_name": fname, "version": "1.0.0",
        "calculation": {"expression": expr, "description": desc},
        "dependencies": deps, "parameters": params,
        "expected_direction": int(np.sign(direction)) if direction != 0 else 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": f"2020-01-01..{WARM_END}",
            "last_validated": "2027-01-19",
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
                "description": f"Factor signal panel: rows = dates, cols = assets. Shape {artifact.shape}.",
                "columns": WATCH, "shape": list(artifact.shape),
                "n_valid_values": int(np.isfinite(artifact).sum()),
                "sha256": str(hash(artifact.tobytes()) & 0xFFFFFFFFFFFFFFFF),
                "data": save_artifact_csv(factor_full),
            },
        },
        "tags": tags,
        "signal_artifact": f"{name}.signal.npy",
        "artifact_provenance": {
            "format": "npy_matrix", "shape": list(artifact.shape), "columns": WATCH,
            "dates_first": str(close.index.min().date()), "dates_last": str(close.index.max().date()),
            "n_nan": int(np.isnan(artifact).sum()),
        },
        "benchmark_admission": {
            "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE, "correlation_threshold": CORR_GATE},
            "selected_metrics": {
                "ic": res["ic_h10"], "icir": res["icir_h10"], "metric_path": "validation.metrics",
                "max_abs_library_correlation": res["max_abs_library_correlation"],
                "correlation_path": "validation.metrics.max_abs_library_correlation",
            },
        },
    }
    path = Path("factors") / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"  PERSISTED -> {path}")
    return path


def verify_factor(name):
    path = Path("factors") / f"{name}.json"
    loaded = json.loads(path.read_text())
    assert loaded["factor_id"] == name, "id mismatch"
    assert loaded["validation"]["status"] == "EFFECTIVE", "status not EFFECTIVE"
    m = loaded["validation"]["metrics"]
    assert abs(m["ic"]) >= IC_GATE, "ic below gate"
    assert abs(m["icir"]) >= ICIR_GATE, "icir below gate"
    assert loaded["validation"]["signal_artifact"]["format"] == "base64:zlib:csv"
    npy = np.load(Path("factors") / f"{name}.signal.npy")
    assert npy.shape == tuple(loaded["artifact_provenance"]["shape"])
    print(f"  VERIFIED {name}: JSON ok, EFFECTIVE, ic={m['ic']:+.4f} icir={m['icir']:+.4f} "
          f"maxcorr={m['max_abs_library_correlation']:.3f} npy={npy.shape}")


if __name__ == "__main__":
    close, open_, high, low, vol, macro = load_data()
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; warm-up ..{WARM_END}; "
          f"data end {close.index[-1].date()}", flush=True)
    libsig = library_signals(close, macro)
    print(f"library factors: {list(libsig.keys())}\n", flush=True)

    print("##### A) LIBRARY RE-VALIDATION (warm-up admission + live drift) #####", flush=True)
    lib_results = {}
    for fid, sig in libsig.items():
        r = validate(f"[LIB] {fid}", sig, close, libsig)
        lib_results[fid] = r
        # informational live drift
        sub = sig.loc[LIVE_START:]
        if sub.notna().sum().sum() < 200:
            print(f"  {fid}: too few live obs, skip drift\n")
            continue
        ic = rank_ic_series(sub, fwd_returns(close, 10).loc[LIVE_START:])
        if len(ic) >= 5:
            d = 1.0 if ic.mean() >= 0 else -1.0
            icir = d * ic.mean() / ic.std() if ic.std() > 0 else float("nan")
            print(f"  {fid}: LIVE h10 IC={d*ic.mean():+.4f} ICIR={icir:+.4f} hit={(d*ic>0).mean():.3f} n={len(ic)}\n")

    print("##### B) ROUND-18 CANDIDATE SCREENS #####", flush=True)
    cands = {
        "rsi_14_skip2": lambda: cand_rsi_14_skip2(close),
        "win_rate_20d_skip5": lambda: cand_win_rate_20d_skip5(close),
        "updown_capture_60": lambda: cand_updown_capture_60(close),
        "zscore_60d": lambda: cand_zscore_60d(close),
        "hl_range_pos_20": lambda: cand_hl_range_pos_20(close),
        "corr_ew_drift_60x20": lambda: cand_corr_ew_drift_60x20(close),
        "skew_20d_skip5": lambda: cand_skew_20d_skip5(close),
        "dd_depth_60": lambda: cand_dd_depth_60(close),
        "vol_zscore_20x120": lambda: cand_vol_zscore_20x120(close),
        "ewma_cross_20x60": lambda: cand_ewma_cross_20x60(close),
        "rates_beta_cond_60x20": lambda: cand_rates_beta_cond_60x20(close, macro),
        "crypto_beta_cond_20": lambda: cand_crypto_beta_cond_20(close),
        "overnight_share_20": lambda: cand_overnight_share_20(close, open_),
        "rel_vol_20": lambda: cand_rel_vol_20(close),
        "mom60_skip5_rel": lambda: cand_mom60_skip5_rel(close),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===\n", flush=True)

    print("##### SUMMARY #####", flush=True)
    for name, r in results.items():
        print(f"{name:<26} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
              f"maxcorr={r['max_abs_library_correlation']:.3f} cov_ge8={r['coverage_dates_ge8']:.2f} "
              f"turn={r['turnover_10d_rank']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}", flush=True)

    with open("scripts/miner_1_20270119_screen_results.json", "w") as f:
        json.dump({"candidates": results, "library": lib_results}, f, indent=1, default=str)
    print("saved -> scripts/miner_1_20270119_screen_results.json", flush=True)
