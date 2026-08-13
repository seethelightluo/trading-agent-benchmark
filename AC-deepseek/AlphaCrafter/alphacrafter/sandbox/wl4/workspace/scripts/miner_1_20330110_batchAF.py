"""miner_1 (2033-01-10): explore batch AF - new cross-asset factor candidates.

Universe: 15 tradable cross-asset instruments (equity idx, commodities, crypto, yields).
Observation-only macro: DXY, USDCNY, USDJPY, EURUSD, VIX (never traded).
Data loaded through previous completed trading day (visible_through=2033-01-07), no lookahead.

Candidates (all new / distinct from library + evicted families):
  A. us10y_beta_60d      - rolling beta of asset ret on US10Y yield change (rate sensitivity)
  B. spx_beta_60d        - rolling beta of asset ret on SPX ret (equity beta)
  C. wti_beta_60d        - rolling beta of asset ret on WTI ret (commodity/oil beta)
  D. mom_confluence_20x60 - min(|m20|,|m60|)*sign(m20)*sign(m60) (2-window momentum agreement)
  E. win_rate_20d        - fraction of positive daily returns over 20d (consistency)
  F. up_vol_ratio_20d    - mean(up-day ret)/std(ret,20) (upside capture per unit risk)
  G. skew_vol_20d        - skewness(ret,20)/std(ret,20) (risk-adjusted skew)
  H. range_vol_ratio_20d - mean((high-low)/close,20)/std(ret,20) (range efficiency)
  I. downside_vol_ratio_20d - std(neg-day ret)/std(all ret) over 20d (downside vol asymmetry)
  J. worst5_vol_20d      - min 5d cumulative ret over past 20d / vol (crash sensitivity)
  K. volume_trend_corr_20d - corr(volume, close) over 20d (volume confirmation)
  L. zscore_120d         - (close - SMA120)/std120 (long-term Bollinger position)
  M. xau_spx_corr_diff_60d - corr(asset,XAU) - corr(asset,SPX) over 60d (safe-haven vs equity affinity)
  N. mom_120d_voladj     - 120d momentum / std(ret,60) (long momentum per risk)

Admission gates (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Also re-validates the 3 library factors for drift tracking.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr,
    library_signals, full_eval,
)

panels = load_panels(days=3200)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape={closes.shape} range={closes.index.min().date()} -> {closes.index.max().date()}")
print(f"assets={closes.shape[1]} (15-asset cross-asset universe)")

lib_sig = library_signals(panels, closes, rets)
H = 10  # admission horizon
fwd = forward_returns(closes, H)

# ---------------- helper: rolling beta ----------------
def rolling_beta(a, b, win, min_obs):
    cov = a.rolling(win, min_periods=min_obs).cov(b)
    var = b.rolling(win, min_periods=min_obs).var()
    return cov / var

def clean(fp):
    return fp.replace([np.inf, -np.inf], np.nan)

us10y_ret = panels["US10Y"]["close"].astype(float).pct_change()
cn10y_ret = panels["CN10Y"]["close"].astype(float).pct_change()
spx_ret = rets["SPX"]
wti_ret = rets["WTI"]
xau_ret = rets["XAU"]

cands = {}

# A. US10Y rate beta 60d
cands["us10y_beta_60d"] = clean(rolling_beta(rets, us10y_ret, 60, 40))
# B. SPX beta 60d
cands["spx_beta_60d"] = clean(rolling_beta(rets, spx_ret, 60, 40))
# C. WTI beta 60d
cands["wti_beta_60d"] = clean(rolling_beta(rets, wti_ret, 60, 40))
# D. momentum confluence 20x60
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
cands["mom_confluence_20x60"] = clean(np.minimum(m20.abs(), m60.abs()) * np.sign(m20) * np.sign(m60))
# E. win rate 20d
cands["win_rate_20d"] = (rets > 0).rolling(20, min_periods=15).mean()
# F. upside capture per unit risk 20d
up = rets.where(rets > 0)
cands["up_vol_ratio_20d"] = clean(up.rolling(20, min_periods=15).mean() / rets.rolling(20, min_periods=15).std())
# G. risk-adjusted skewness 20d
cands["skew_vol_20d"] = clean(rets.rolling(20, min_periods=15).skew() / rets.rolling(20, min_periods=15).std())
# H. range efficiency 20d
hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).sort_index()
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).sort_index()
rng = ((hi - lo) / closes).replace([np.inf, -np.inf], np.nan)
cands["range_vol_ratio_20d"] = clean(rng.rolling(20, min_periods=15).mean() / rets.rolling(20, min_periods=15).std())
# I. downside vol asymmetry 20d
neg = rets.where(rets < 0)
cands["downside_vol_ratio_20d"] = clean(neg.rolling(20, min_periods=15).std() / rets.rolling(20, min_periods=15).std())
# J. worst 5d window over 20d, scaled by vol
worst5 = closes.rolling(20, min_periods=15).apply(
    lambda x: (x / x.shift(5) - 1.0).min() if len(x) >= 10 else np.nan, raw=False)
cands["worst5_vol_20d"] = clean(worst5 / rets.rolling(20, min_periods=15).std())
# K. volume-trend correlation 20d
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()
vt = {}
for a in closes.columns:
    z = pd.concat([vol_panel[a].rename("v"), closes[a].rename("c")], axis=1)
    vt[a] = z.rolling(20, min_periods=15).corr()
vt = pd.DataFrame(vt, index=closes.index)
cands["volume_trend_corr_20d"] = clean(vt)
# L. z-score 120d
cands["zscore_120d"] = clean((closes - closes.rolling(120).mean()) / closes.rolling(120).std())
# M. safe-haven vs equity affinity 60d
xau_corr = rets.rolling(60, min_periods=40).corr(xau_ret)
spx_corr = rets.rolling(60, min_periods=40).corr(spx_ret)
cands["xau_spx_corr_diff_60d"] = clean(xau_corr - spx_corr)
# N. 120d momentum / vol60
cands["mom_120d_voladj"] = clean(m60 / rets.rolling(60, min_periods=40).std())

# ---------------- evaluate ----------------
print("\n=== LIBRARY FACTORS (drift re-validation, h=10) ===")
for name, fp in lib_sig.items():
    m, ics = full_eval(fp, closes, library=lib_sig, expected_sign=1)
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']} cov_days={m['coverage_asset_days']:.2f} cov8={m['coverage_dates_ge8']:.2f} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')}")

print("\n=== CANDIDATE FACTORS (h=10, gates |IC|>=0.0070, |ICIR|>=0.0840) ===")
rows = []
for name, fp in cands.items():
    ics = rank_ic_series(fp, fwd, min_valid=8)
    if len(ics) < 30:
        print(f"{name:26s} INSUFFICIENT dates ({len(ics)})")
        continue
    m = summarize_ic(ics, expected_sign=1)
    m.update(coverage_metrics(fp, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(fp, 10)
    m["decay_ic_by_horizon"] = decay_profile(fp, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(fp, lib_sig)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    sign = "+" if m["ic"] >= 0 else "-"
    gate = "PASS" if (abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840) else "FAIL"
    rows.append((name, m, gate, sign))
    dec = m["decay_ic_by_horizon"]
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={m['n_ic_dates']} "
          f"cov_days={m['coverage_asset_days']:.2f} cov8={m['coverage_dates_ge8']:.2f} turn={m['turnover_10d_rank']} "
          f"maxcorr={corr:.3f}({key}) decay1/10/20={dec.get('1')}/{dec.get('10')}/{dec.get('20')} => {gate}")

print("\n=== PASS SUMMARY ===")
for name, m, gate, sign in rows:
    if gate == "PASS":
        print(f"{name} dir={sign}1 IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} maxcorr={m['max_abs_library_correlation']:.3f}")
