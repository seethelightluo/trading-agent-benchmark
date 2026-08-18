"""miner_1 cycle 21: screen a new batch of novel, library-orthogonal factor families.

Families NOT currently in library (mom/rel-mom/vol-of-vol/max-ret/downside-vol/
beta-ew/vix-beta/vol-adj-mom/amihud/eurusd-beta): realized skew/kurtosis,
variance ratio (trend persistence), efficiency/range position, drawdown depth,
volume-pressure (OBV), price-volume correlation, RSI, overnight vs intraday
momentum, up/down vol ratio, Garman-Klass efficiency, pairwise comovement,
rate-spread (US10Y-CN10Y) conditional beta, USDJPY conditional beta.

Admission gates (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 at h=10,
max_abs_library_correlation < 0.5 (self-reported provenance only).
Validation window: 2020-01-01..2026-07-15 (research warm-up), same as library.
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_fastlib import (  # noqa: E402
    WATCH, load_panel, load_ohlc_volume, load_macro, fwd_returns,
    library_signals, lib_corr_fast, validate_fast, HORIZONS, FACTOR_LAST, EPS,
)

panel = load_panel()
rets = panel.pct_change()
mkt_ret = panel.mean(axis=1).pct_change()
macr = load_macro()
libs = library_signals(panel)
fwd = {h: fwd_returns(panel, h) for h in HORIZONS}
fwd_rank_cache = {h: fwd[h].loc[:FACTOR_LAST].rank(axis=1).values.astype(float) for h in HORIZONS}
OHLC = load_ohlc_volume()
print(f"panel {panel.shape}, factor window .. {FACTOR_LAST}", flush=True)

C = {}


def per_asset_series(fn):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = fn(s)
    return pd.DataFrame(cols, index=panel.index)


# ---- 1. corr_median_60 (re-validation; PASS in cycle 19, not persisted yet) ----
def cand_corr_median_60():
    med_r = rets.median(axis=1)
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        mr = med_r.reindex(s.index)
        z = pd.concat([s.pct_change().rename("r"), mr.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60, min_periods=30).corr(z["m"])
    return pd.DataFrame(cols, index=panel.index)


C["corr_median_60"] = cand_corr_median_60

# ---- 2. pairwise comovement: mean rolling corr with all other assets ----
def cand_corr_ew_60():
    cols = {}
    rmat = rets.values.astype(float)
    idx = panel.index
    for j, a in enumerate(panel.columns):
        x = rmat[:, j]
        others = np.delete(rmat, j, axis=1)
        # rolling 60d pairwise corr vectorized per asset
        out = np.full(len(x), np.nan)
        for i in range(60, len(x)):
            w = slice(i - 60, i)
            xw = x[w]
            ow = others[w]
            valid = np.isfinite(xw)[:, None] & np.isfinite(ow)
            if valid.sum() < 30:
                continue
            xc = xw - np.nanmean(xw)
            oc = ow - np.nanmean(ow, axis=0)
            num = np.nansum(xc[:, None] * oc, axis=0)
            dx = np.sqrt(np.nansum(xc * xc))
            do = np.sqrt(np.nansum(oc * oc, axis=0))
            with np.errstate(invalid="ignore"):
                r = num / (dx * do)
            out[i] = np.nanmean(r[np.isfinite(r)])
        cols[a] = pd.Series(out, index=idx)
    return pd.DataFrame(cols, index=idx)


C["corr_ew_60"] = cand_corr_ew_60

# ---- 3. realized skewness 60d ----
def cand_skewness_60():
    mu = rets.rolling(60, min_periods=30).mean()
    sd = rets.rolling(60, min_periods=30).std()
    return ((rets - mu) ** 3).rolling(60, min_periods=30).mean() / (sd ** 3 + EPS)


C["skewness_60"] = cand_skewness_60

# ---- 4. realized kurtosis 60d ----
def cand_kurtosis_60():
    mu = rets.rolling(60, min_periods=30).mean()
    sd = rets.rolling(60, min_periods=30).std()
    return ((rets - mu) ** 4).rolling(60, min_periods=30).mean() / (sd ** 4 + EPS) - 3.0


C["kurtosis_60"] = cand_kurtosis_60

# ---- 5. variance ratio 5x60 (trend persistence; >1 trending) ----
def cand_variance_ratio_5x60():
    v5 = rets.rolling(5, min_periods=3).var() * 12.0
    v60 = rets.rolling(60, min_periods=30).var()
    return v5 / (v60 + EPS)


C["variance_ratio_5x60"] = cand_variance_ratio_5x60

# ---- 6. Kaufman efficiency ratio 20d ----
def cand_trend_eff_ratio_20():
    def _er(s):
        n = 20
        change = s.diff(n).abs()
        vol = s.diff().abs().rolling(n, min_periods=10).sum()
        return change / (vol + EPS)
    return per_asset_series(_er)


C["trend_eff_ratio_20"] = cand_trend_eff_ratio_20

# ---- 7. range position 60d: (close - min60)/(max60 - min60) ----
def cand_range_pos_60():
    def _rp(s):
        n = 60
        hi = s.rolling(n, min_periods=30).max()
        lo = s.rolling(n, min_periods=30).min()
        return (s - lo) / (hi - lo + EPS)
    return per_asset_series(_rp)


C["range_pos_60"] = cand_range_pos_60

# ---- 8. stochastic intraday position 20d: mean((close-low)/(high-low)) ----
def cand_hl_pos_20():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        pos = (df["close"] - df["low"]) / (df["high"] - df["low"] + EPS)
        cols[a] = pos.rolling(20, min_periods=10).mean().reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["hl_pos_20"] = cand_hl_pos_20

# ---- 9. drawdown depth 60d (positive = deeper underwater) ----
def cand_dd_depth_60():
    def _dd(s):
        roll_max = s.rolling(60, min_periods=30).max()
        return -(s / roll_max - 1.0)
    return per_asset_series(_dd)


C["dd_depth_60"] = cand_dd_depth_60

# ---- 10. run streak: consecutive up-day count ----
def cand_run_streak():
    def _streak(s):
        up = (s.diff() > 0).astype(float)
        out = np.zeros(len(s))
        cnt = 0.0
        for i in range(len(s)):
            cnt = cnt + 1.0 if up.iloc[i] else 0.0
            out[i] = cnt
        return pd.Series(out, index=s.index)
    return per_asset_series(_streak)


C["run_streak"] = cand_run_streak

# ---- 11. up/down vol ratio 20d (positive = upside vol dominates) ----
def cand_up_vol_ratio_20():
    up = rets.clip(lower=0)
    dn = rets.clip(upper=0)
    upv = (up ** 2).rolling(20, min_periods=10).mean()
    dnv = (dn ** 2).rolling(20, min_periods=10).mean()
    return np.sqrt(upv) / (np.sqrt(dnv) + EPS)


C["up_vol_ratio_20"] = cand_up_vol_ratio_20

# ---- 12. win rate 20d ----
def cand_win_rate_20():
    return (rets > 0).rolling(20, min_periods=10).mean()


C["win_rate_20"] = cand_win_rate_20

# ---- 13. RSI 14 (simple) ----
def cand_rsi_14():
    gain = rets.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-rets.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / (loss + EPS)
    return rs / (1.0 + rs)


C["rsi_14"] = cand_rsi_14

# ---- 14. OBV slope 20d (normalized money flow) ----
def cand_obv_slope_20():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        r = df["close"].pct_change()
        v = df["volume"]
        signed = np.sign(r) * v
        obv = signed.rolling(20, min_periods=10).sum()
        norm = v.rolling(20, min_periods=10).mean()
        cols[a] = (obv / (norm + EPS)).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["obv_slope_20"] = cand_obv_slope_20

# ---- 15. price-volume correlation 20d ----
def cand_price_vol_corr_20():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        r = df["close"].pct_change()
        dv = df["volume"].pct_change()
        cols[a] = r.rolling(20, min_periods=10).corr(dv).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["price_vol_corr_20"] = cand_price_vol_corr_20

# ---- 16. Garman-Klass / close-close vol ratio 60d ----
def cand_gk_ratio_60():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        o, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
        gk = 0.5 * (np.log(hi / lo) ** 2) - (2 * np.log(2) - 1) * (np.log(cl / o) ** 2)
        gk_vol = np.sqrt(gk.rolling(60, min_periods=30).mean())
        cc_vol = cl.pct_change().rolling(60, min_periods=30).std()
        cols[a] = (gk_vol / (cc_vol + EPS)).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["gk_ratio_60"] = cand_gk_ratio_60

# ---- 17. overnight momentum 20d (cum open/prev_close - 1) ----
def cand_overnight_mom_20():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        ov = df["open"] / df["close"].shift(1) - 1.0
        cols[a] = ov.rolling(20, min_periods=10).sum().reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["overnight_mom_20"] = cand_overnight_mom_20

# ---- 18. intraday momentum 20d (cum close/open - 1) ----
def cand_intraday_mom_20():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        idr = df["close"] / df["open"] - 1.0
        cols[a] = idr.rolling(20, min_periods=10).sum().reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["intraday_mom_20"] = cand_intraday_mom_20

# ---- 19. US10Y-CN10Y spread beta conditional on spread momentum ----
def cand_rate_spread_beta_cond(beta_win=60, fx_win=20):
    spread = panel["US10Y"] - panel["CN10Y"]
    spr_ret = spread.pct_change()
    spr_mom = spread / spread.shift(fx_win) - 1.0
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        own = s.index
        sr = spr_ret.reindex(own).ffill()
        sm = spr_mom.reindex(own).ffill()
        r = s.pct_change()
        b = r.rolling(beta_win, min_periods=30).cov(sr) / sr.rolling(beta_win, min_periods=30).var().replace(0, np.nan)
        cols[a] = (b * sm).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["rate_spread_beta_cond_60x20"] = cand_rate_spread_beta_cond

# ---- 20. USDJPY beta conditional on USDJPY momentum (yen carry regime) ----
def cand_usdjpy_beta_cond(beta_win=60, fx_win=20):
    jpy = macr["USDJPY"]
    jpy_ret = jpy.pct_change()
    jpy_mom = jpy / jpy.shift(fx_win) - 1.0
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        own = s.index
        jr = jpy_ret.reindex(own).ffill()
        jm = jpy_mom.reindex(own).ffill()
        r = s.pct_change()
        b = r.rolling(beta_win, min_periods=30).cov(jr) / jr.rolling(beta_win, min_periods=30).var().replace(0, np.nan)
        cols[a] = (b * jm).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["usdjpy_beta_cond_60x20"] = cand_usdjpy_beta_cond

# ---- 21. idiosyncratic momentum: 20d mom minus EW market mom ----
def cand_idio_mom_20x60():
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    mkt_mom = m20.mean(axis=1)
    v = rets.rolling(60, min_periods=30).std()
    resid = m20.sub(mkt_mom, axis=0)
    return resid / (v + EPS)


C["idio_mom_20x60"] = cand_idio_mom_20x60

# ---- 22. Parkinson vol ratio 20d ----
def cand_parkinson_ratio_20():
    cols = {}
    for a in panel.columns:
        df = OHLC[a]
        hi, lo, cl = df["high"], df["low"], df["close"]
        park = np.sqrt((np.log(hi / lo) ** 2) / (4 * np.log(2)))
        park_vol = park.rolling(20, min_periods=10).mean()
        cc_vol = cl.pct_change().rolling(20, min_periods=10).std()
        cols[a] = (park_vol / (cc_vol + EPS)).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["parkinson_ratio_20"] = cand_parkinson_ratio_20

# ---- run validation ----
results = {}
for name, fn in C.items():
    try:
        factor = fn()
        factor = factor.reindex(panel.index)
        res = validate_fast(name, factor, panel, fwd, libs, fwd_rank_cache)
        results[name] = res
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===", flush=True)
        results[name] = {"name": name, "error": str(e)}

print("\n===== SUMMARY (h10) =====", flush=True)
for name, r in results.items():
    if "error" in r:
        print(f"{name:<30} ERROR {r['error']}", flush=True)
        continue
    g = r["admission_gate"]
    print(f"{name:<30} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"hit={r['hit_h10']:.3f} turn={r['turnover_10d_rank']:.2f} "
          f"cov={r['coverage_asset_days']:.3f} maxcorr={r['max_abs_library_correlation']} "
          f"-> {'PASS' if g['pass'] else 'FAIL'}", flush=True)

with open("scripts/miner_1_cycle21_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved -> scripts/miner_1_cycle21_results.json", flush=True)
