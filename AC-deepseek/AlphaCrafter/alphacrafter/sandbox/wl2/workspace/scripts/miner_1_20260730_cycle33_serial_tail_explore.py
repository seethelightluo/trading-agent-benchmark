"""miner_1 cycle 33: explore SERIAL-DEPENDENCE / TAIL / PATH-STRUCTURE factors.

Motivation: active library is strong in momentum (mom20_volproxy60, mom30_vol60,
mom_*_skip5), volatility (vol_of_vol20x60, volcluster_60, calmness_20) and macro-beta
(dxy/usdjpy/vix/downbeta/lagbeta). Cycle32(b) covered driver-beta and volume/OHLC.
This cycle targets orthogonal dimensions NOT yet in the library:
  1. autocorr_60       - lag-1 daily-return autocorrelation (serial dependence)
  2. parkinson_ratio_20x60 - Parkinson (high-low) vol vs close-close vol (estimator divergence)
  3. semi_vol_asym_60  - downside semi-dev / upside semi-dev (volatility asymmetry)
  4. win_rate_60       - fraction of positive days over 60d (trend consistency)
  5. ew_beta_60        - rolling beta vs equal-weight cross-asset basket (systematic beta)
  6. kurtosis_60       - excess kurtosis of daily returns over 60d (tail fatness)
  7. trend_r2_60       - R^2 of log-price linear trend over 60d (trend quality)
  8. close_loc_asym_60 - mean (high-close)/(close-low) ratio (intraday direction asymmetry)
  9. corr_ew_60        - average pairwise 60d return correlation vs other 14 assets
 10. drawdown_252      - (close / rolling_max252 - 1) long-horizon drawdown depth

Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation<0.5, turnover vs 10d cadence.
Validation date 2026-07-30 (data visible through 2026-07-29). No lookahead.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_asset, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor, turnover_rank)

panel = load_panel()
union_idx = panel.index

frames = {a: load_asset(a) for a in TRADABLES}

def own_series(a, col):
    s = frames[a][col].astype(float)
    s.index = pd.to_datetime(frames[a]["date"].values)
    return s

def reindex_to_panel(series_dict):
    out = {}
    for a, s in series_dict.items():
        s.index = pd.to_datetime(s.index)
        out[a] = s.reindex(union_idx)
    return pd.DataFrame(out, index=union_idx)

cands = {}

# 1. lag-1 autocorrelation of daily returns over 60d
ac = {}
for a in TRADABLES:
    c = own_series(a, "close")
    r = c.pct_change()
    ac[a] = r.rolling(60, min_periods=30).apply(
        lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) >= 30 and np.nanstd(x) > 0 else np.nan,
        raw=True)
cands["autocorr_60"] = reindex_to_panel(ac)

# 2. Parkinson vol ratio: mean((high-low)/close,20) / close-close 20d std
pr = {}
for a in TRADABLES:
    c = own_series(a, "close")
    hi, lo = own_series(a, "high"), own_series(a, "low")
    rng = (hi - lo) / c
    cvol = c.pct_change().rolling(20, min_periods=10).std()
    pr[a] = rng.rolling(20, min_periods=10).mean() / (cvol + 1e-12)
cands["parkinson_ratio_20x60"] = reindex_to_panel(pr)

# 3. downside semi-dev / upside semi-dev over 60d
sa = {}
for a in TRADABLES:
    c = own_series(a, "close")
    r = c.pct_change()
    dd = r.where(r < 0)
    ud = r.where(r > 0)
    dstd = dd.rolling(60, min_periods=30).std()
    ustd = ud.rolling(60, min_periods=30).std()
    sa[a] = dstd / (ustd + 1e-12)
cands["semi_vol_asym_60"] = reindex_to_panel(sa)

# 4. win rate: fraction of positive days over 60d
wr = {}
for a in TRADABLES:
    c = own_series(a, "close")
    r = c.pct_change()
    wr[a] = (r > 0).rolling(60, min_periods=30).mean()
cands["win_rate_60"] = reindex_to_panel(wr)

# 5. rolling beta vs equal-weight cross-asset basket (60d)
ret_panel = panel.pct_change()
ew_ret = ret_panel.mean(axis=1)  # union-index EW basket return
eb = {}
for a in TRADABLES:
    c = own_series(a, "close")
    ar = c.pct_change()
    br = ew_ret.reindex(ar.index).ffill()
    df = pd.concat([ar.rename("a"), br.rename("b")], axis=1).dropna()
    eb[a] = df["a"].rolling(60, min_periods=30).cov(df["b"]) / (df["b"].rolling(60, min_periods=30).var() + 1e-12)
cands["ew_beta_60"] = reindex_to_panel(eb)

# 6. excess kurtosis of daily returns over 60d
ku = {}
for a in TRADABLES:
    c = own_series(a, "close")
    ku[a] = c.pct_change().rolling(60, min_periods=30).kurt()
cands["kurtosis_60"] = reindex_to_panel(ku)

# 7. R^2 of log-price linear trend over 60d (trend quality)
def trend_r2(s):
    lp = np.log(s)
    def f(x):
        t = np.arange(len(x))
        if np.std(x) == 0:
            return np.nan
        return float(np.corrcoef(t, x)[0, 1] ** 2)
    return lp.rolling(60, min_periods=30).apply(f, raw=True)
cands["trend_r2_60"] = per_asset(panel, trend_r2)

# 8. intraday direction asymmetry: mean((high-close)/(close-low), 60d)
cla = {}
for a in TRADABLES:
    c = own_series(a, "close")
    hi, lo = own_series(a, "high"), own_series(a, "low")
    ratio = (hi - c) / ((c - lo) + 1e-12)
    cla[a] = ratio.rolling(60, min_periods=30).mean()
cands["close_loc_asym_60"] = reindex_to_panel(cla)

# 9. average pairwise 60d return correlation vs other 14 assets
def corr_ew(s, ret_panel_all):
    ar = s.pct_change()
    out = {}
    for other in ret_panel_all.columns:
        o = ret_panel_all[other].reindex(ar.index)
        df = pd.concat([ar.rename("a"), o.rename("o")], axis=1).dropna()
        out[other] = df["a"].rolling(60, min_periods=30).corr(df["o"])
    corr_df = pd.DataFrame(out, index=ar.index)
    return corr_df.mean(axis=1)
ce = {a: corr_ew(own_series(a, "close"), ret_panel) for a in TRADABLES}
cands["corr_ew_60"] = reindex_to_panel(ce)

# 10. long-horizon drawdown depth from 252d high
dd = {}
for a in TRADABLES:
    c = own_series(a, "close")
    dd[a] = c / c.rolling(252, min_periods=120).max() - 1.0
cands["drawdown_252"] = reindex_to_panel(dd)

# ---------------------------------------------------------------------------
# Library signals (active/reference set) for correlation gate
# ---------------------------------------------------------------------------
close = panel
lib = {}
lib["mom20_volproxy60"] = per_asset(close, lambda s: (s.shift(5) / s.shift(25) - 1.0) / (1.0 + (s.shift(5) / s.shift(65) - 1.0).abs()))
def calmness_20(s):
    return s.pct_change().abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan, raw=True)
lib["calmness_20"] = per_asset(close, calmness_20)
lib["vol_of_vol20x60"] = per_asset(close, lambda s: s.pct_change().rolling(20).std().rolling(60).std())
lib["volcluster_60"] = per_asset(close, lambda s: (s.pct_change().rolling(5).std() / s.pct_change().rolling(60).std()).rolling(60).mean())
def gain_loss_20(s):
    r = s.pct_change()
    return r.clip(lower=0).rolling(20, min_periods=10).sum() / (r.clip(upper=0).abs().rolling(20, min_periods=10).sum() + 1e-12)
lib["gain_loss_20"] = per_asset(close, gain_loss_20)
def max_consec_gain_20(s):
    r = (s.pct_change() > 0).astype(float)
    def f(x):
        m, cur = 0, 0
        for v in x:
            cur = cur + 1 if v > 0 else 0
            m = max(m, cur)
        return float(m)
    return r.rolling(20, min_periods=10).apply(f, raw=True)
lib["max_consec_gain_20"] = per_asset(close, max_consec_gain_20)
lib["range_pos_252"] = per_asset(close, lambda s: (s - s.rolling(252, min_periods=120).min()) / (s.rolling(252, min_periods=120).max() - s.rolling(252, min_periods=120).min() + 1e-12))
def days_since_high_60(s):
    rolling_max = s.rolling(60, min_periods=30).max()
    is_high = (s == rolling_max).astype(float)
    count = 0
    out = []
    vals = is_high.values
    for v in vals:
        count = 0 if v == 1 else count + 1
        out.append(count)
    return pd.Series(out, index=s.index)
lib["days_since_high_60"] = per_asset(close, days_since_high_60)

def beta_cond(asset_close, driver_close, w=60, m=20):
    dcs = driver_close.reindex(asset_close.index).ffill()
    ar, dr = asset_close.pct_change(), dcs.pct_change()
    df = pd.concat([ar.rename("a"), dr.rename("d")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=max(int(w*0.5), 15)).cov(df["d"]) / (df["d"].rolling(w, min_periods=max(int(w*0.5), 15)).var() + 1e-12)
    return b * (dcs / dcs.shift(m) - 1.0).reindex(b.index)
dxy = macro_series("DXY")
usdjpy = macro_series("USDJPY")
vix = macro_series("VIX")
lib["dxy_beta_cond_60x20"] = per_asset(close, beta_cond, dxy, 60, 20)
lib["usdjpy_beta_cond_120x60"] = per_asset(close, beta_cond, usdjpy, 120, 60)
lib["vix_beta_cond_60x20"] = per_asset(close, beta_cond, vix, 60, 20)
spx = macro_series("SPX")
def downbeta_spx_60(s):
    sp = spx.reindex(s.index).ffill()
    ar, sr = s.pct_change(), sp.pct_change()
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    neg = df[df["s"] < 0]
    return neg["a"].rolling(60, min_periods=15).cov(neg["s"]) / (neg["s"].rolling(60, min_periods=15).var() + 1e-12)
lib["downbeta_spx_60"] = per_asset(close, downbeta_spx_60)
def lagbeta_spx_60(s):
    sp = spx.reindex(s.index).ffill()
    ar, sr = s.pct_change(), sp.pct_change().shift(1)
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    return df["a"].rolling(60, min_periods=15).cov(df["s"]) / (df["s"].rolling(60, min_periods=15).var() + 1e-12)
lib["lagbeta_spx_60"] = per_asset(close, lagbeta_spx_60)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
fwd_cache = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd_cache[str(h)] = forward_returns(panel, h)

print("=" * 100)
print("CYCLE 33 SERIAL/TAIL/PATH-STRUCTURE EXPLORATION - validation date 2026-07-30")
print("panel dates: %s .. %s  (n=%d)  assets=%d" % (union_idx.min().date(), union_idx.max().date(), len(union_idx), len(TRADABLES)))
print("=" * 100)

results = {}
for name, sig in cands.items():
    m = validate_factor(sig, panel, library=lib, fwd_cache=fwd_cache)
    results[name] = m
    ic = abs(m["ic"]); icir = abs(m["icir"])
    passed = (ic >= 0.007) and (icir >= 0.084)
    print("[%s] IC=%s ICIR=%s hit=%s n=%s cov_asset=%s cov_dates=%s turn=%s maxlib=%s => %s"
          % (name, m["ic"], m["icir"], m.get("ic_hit_ratio"), m.get("n_ic_dates"),
             m.get("coverage_asset_days"), m.get("coverage_dates_ge8"),
             m.get("turnover_10d_rank"), m.get("max_abs_library_correlation"),
             "PASS" if passed else "FAIL"))
    if m.get("library_pairwise_corr"):
        top = sorted(m["library_pairwise_corr"].items(), key=lambda kv: -abs(kv[1]))[:3]
        print("     top-lib-corr:", [(k, v) for k, v in top])

print("=" * 100)
print("DECAY TABLES (passers & near-passers):")
for name in results:
    m = results[name]
    if abs(m["ic"]) >= 0.006 or abs(m["icir"]) >= 0.07:
        print(name, json.dumps(m["decay_ic_by_horizon"]))
