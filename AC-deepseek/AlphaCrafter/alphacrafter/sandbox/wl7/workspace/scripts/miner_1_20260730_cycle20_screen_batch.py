"""miner_1 cycle 20: re-validate corr_median_60 (passed gate in cycle 19, not
persisted) + screen a new batch of novel, library-orthogonal factor families.

Universe: 15 tradable cross-asset instruments. Admission gates (benchmark
contract): |IC|>=0.007 and |ICIR|>=0.084 at h=10, max_abs_library_correlation<0.5.
Uses the shared vectorized miner_1_fastlib for IC/ICIR/turnover/corr/decay.
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_fastlib import (  # noqa: E402
    WATCH, load_panel, load_ohlc_volume, load_macro, fwd_returns, rank_ic_fast,
    turnover_10d_rank_fast, library_signals, lib_corr_fast, validate_fast,
    HORIZONS, FACTOR_LAST, EPS,
)

panel = load_panel()
rets = panel.pct_change()
mkt_ret = panel.mean(axis=1).pct_change()
macr = load_macro()
libs = library_signals(panel)
fwd = {h: fwd_returns(panel, h) for h in HORIZONS}
fwd_rank_cache = {h: fwd[h].rank(axis=1).values.astype(float) for h in HORIZONS}
print(f"panel {panel.shape}, factor window .. {FACTOR_LAST}", flush=True)

C = {}


def per_asset_series(fn):
    """Apply per-asset Series fn on each asset's own calendar, reindex to union."""
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = fn(s)
    return pd.DataFrame(cols, index=panel.index)


# ---- 1. corr_median_60 (re-validation; cycle 19 PASS) ----
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

# ---- 2. Parkinson vol / close-close vol ratio (intraday range efficiency) ----
OHLC_CACHE = None
def _ohlc():
    global OHLC_CACHE
    if OHLC_CACHE is None:
        OHLC_CACHE = load_ohlc_volume()
    return OHLC_CACHE

def cand_parkinson_ratio(n=20):
    ohlc = _ohlc()
    cols = {}
    for a in panel.columns:
        df = ohlc[a]
        hi, lo, cl = df["high"], df["low"], df["close"]
        park = np.sqrt((np.log(hi / lo) ** 2) / (4 * np.log(2)))
        park_vol = park.rolling(n, min_periods=10).mean()
        cc_vol = cl.pct_change().rolling(n, min_periods=10).std()
        cols[a] = (park_vol / (cc_vol + EPS)).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["parkinson_ratio_20"] = lambda: cand_parkinson_ratio(20)

# ---- 3. max drawdown depth over 60d (underwater severity, negated) ----
def cand_dd_depth(n=60):
    def _dd(s):
        roll_max = s.rolling(n, min_periods=30).max()
        return -(s / roll_max - 1.0)  # positive = deeper drawdown
    return per_asset_series(_dd)


C["dd_depth_60"] = cand_dd_depth

# ---- 4. run-streak momentum: count of consecutive positive daily returns ----
def cand_run_streak():
    def _streak(s):
        up = (s.diff() > 0).astype(float)
        out = np.zeros(len(s))
        cnt = 0.0
        for i in range(len(s)):
            if up.iloc[i]:
                cnt += 1.0
            else:
                cnt = 0.0
            out[i] = cnt
        return pd.Series(out, index=s.index)
    return per_asset_series(_streak)


C["run_streak"] = cand_run_streak

# ---- 5. conditional beta to US10Y-CN10Y spread momentum (rate regime) ----
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


C["rate_spread_beta_cond_60x20"] = lambda: cand_rate_spread_beta_cond(60, 20)

# ---- 6. OBV (on-balance volume) 20d slope normalized ----
def cand_obv_slope(n=20):
    ohlc = _ohlc()
    cols = {}
    for a in panel.columns:
        df = ohlc[a]
        cl = df["close"]
        vol = df["volume"].replace(0, np.nan)
        chg = cl.diff()
        obv = (np.sign(chg) * vol).cumsum()
        obv_slope = obv.diff(n) / (obv.abs().rolling(n, min_periods=10).mean() + EPS)
        cols[a] = obv_slope.reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["obv_slope_20"] = lambda: cand_obv_slope(20)

# ---- 7. position within 60d high-low range ----
def cand_range_pos(n=60):
    def _pos(s):
        hi = s.rolling(n, min_periods=30).max()
        lo = s.rolling(n, min_periods=30).min()
        return (s - lo) / (hi - lo + EPS)
    return per_asset_series(_pos)


C["range_pos_60"] = cand_range_pos

# ---- 8. upside vol ratio: std of positive returns / total vol ----
def cand_up_vol_ratio(n=20):
    up = rets.clip(lower=0)
    up_std = up.rolling(n, min_periods=10).std()
    tot_std = rets.rolling(n, min_periods=10).std()
    return up_std / (tot_std + EPS)


C["up_vol_ratio_20"] = cand_up_vol_ratio

# ---- 9. excess kurtosis of returns over 60d ----
def cand_kurtosis_60():
    return rets.rolling(60, min_periods=30).kurt()


C["kurtosis_60"] = cand_kurtosis_60

# ---- 10. idiosyncratic momentum: 20d mom residual vs EW-market beta ----
def cand_idio_mom_20x60():
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    cov = rets.rolling(60, min_periods=30).cov(mkt_ret)
    var = mkt_ret.rolling(60, min_periods=30).var().replace(0, np.nan)
    beta = cov / var
    mkt_20 = mkt_ret.rolling(20).mean()
    return m20 - beta * mkt_20


C["idio_mom_20x60"] = cand_idio_mom_20x60

# ---- 11. correlation with EW-market return (vs beta_ew: normalized exposure) ----
def cand_corr_ew_60():
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        mr = mkt_ret.reindex(s.index)
        z = pd.concat([s.pct_change().rename("r"), mr.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60, min_periods=30).corr(z["m"])
    return pd.DataFrame(cols, index=panel.index)


C["corr_ew_60"] = cand_corr_ew_60

# ---- 12. cross-sectional z-score of 20d momentum ----
def cand_zscore_mom_20():
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    z = m20.sub(m20.mean(axis=1), axis=0).div(m20.std(axis=1) + EPS, axis=0)
    return z


C["zscore_mom_20"] = cand_zscore_mom_20

# ---- 13. win rate: fraction of up days over 20d ----
def cand_win_rate_20():
    return (rets > 0).rolling(20, min_periods=10).mean()


C["win_rate_20"] = cand_win_rate_20

# ---- 14. Garman-Klass vol vs close-close vol (60d) ----
def cand_gk_ratio_60():
    ohlc = _ohlc()
    cols = {}
    for a in panel.columns:
        df = ohlc[a]
        o, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
        gk = 0.5 * (np.log(hi / lo) ** 2) - (2 * np.log(2) - 1) * (np.log(cl / o) ** 2)
        gk_vol = np.sqrt(gk.rolling(60, min_periods=30).mean())
        cc_vol = cl.pct_change().rolling(60, min_periods=30).std()
        cols[a] = (gk_vol / (cc_vol + EPS)).reindex(panel.index)
    return pd.DataFrame(cols, index=panel.index)


C["gk_ratio_60"] = cand_gk_ratio_60

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

with open("scripts/miner_1_cycle20_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved -> scripts/miner_1_cycle20_results.json", flush=True)
