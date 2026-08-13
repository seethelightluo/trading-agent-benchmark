"""miner_1 screening (2032-09-20): explore novel factor candidates for the 15-asset cross-asset universe.

Candidates (all distinct from existing library families momentum/vol/beta-to-mkt):
  A. trend_eff_60d  : Kaufman efficiency ratio |P_t - P_{t-60}| / sum(|ret|,60)  (trend consistency)
  B. skew_60d       : rolling skewness of daily returns over 60d (crash-risk)
  C. updown_cap_60d : mean(up-day ret)/|mean(down-day ret)| over 60d (capture asymmetry)
  D. spread_beta_60d: beta(asset_ret, pct_change(US10Y - CN10Y), 60) (global yield-spread sensitivity)
  E. autocorr_20d   : 1-day return autocorrelation over 20d (mean-reversion tendency)
  F. zscore_60d     : (close - SMA60)/std60  (Bollinger position)
  G. tail_asym_60d  : max(60d) gain vs max(60d) loss asymmetry
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr,
    library_signals, TRADABLE,
)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape={closes.shape} range={closes.index.min().date()} -> {closes.index.max().date()}")
print(f"assets={closes.shape[1]}")

lib_sig = library_signals(panels, closes, rets)

H = 10  # admission horizon
fwd = forward_returns(closes, H)

cands = {}

# A. Kaufman efficiency ratio 60d
num = (closes - closes.shift(60)).abs()
den = rets.abs().rolling(60).sum()
cands["trend_eff_60d"] = num / den

# B. rolling skewness 60d
cands["skew_60d"] = rets.rolling(60).skew()

# C. up/down capture asymmetry 60d
up = rets.where(rets > 0)
dn = rets.where(rets < 0)
up_mean = up.rolling(60).mean()
dn_mean = dn.rolling(60).mean()
cands["updown_cap_60d"] = up_mean / dn_mean.abs()

# D. beta to US10Y-CN10Y spread change
us10 = panels["US10Y"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
spread = us10 - cn10
spread_ret = spread.pct_change()
beta = {}
for a in closes.columns:
    z = pd.concat([rets[a].rename("a"), spread_ret.rename("s")], axis=1).dropna()
    b = z["a"].rolling(60).cov(z["s"]) / z["s"].rolling(60).var()
    beta[a] = b
cands["spread_beta_60d"] = pd.DataFrame(beta, index=closes.index)

# E. 1-day return autocorrelation over 20d
cands["autocorr_20d"] = rets.rolling(20).apply(lambda x: x.iloc[:-1].corr(x.iloc[1:]) if x.iloc[:-1].std() > 0 and x.iloc[1:].std() > 0 else np.nan, raw=False)

# F. Bollinger z-score 60d
cands["zscore_60d"] = (closes - closes.rolling(60).mean()) / closes.rolling(60).std()

# G. tail asymmetry 60d: max positive 1d return vs min negative over 60d
cands["tail_asym_60d"] = rets.rolling(60).max() / rets.rolling(60).min().abs()

rows = []
for name, fp in cands.items():
    ics = rank_ic_series(fp, fwd, min_valid=8)
    m = summarize_ic(ics, expected_sign=1)
    m.update(coverage_metrics(fp, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(fp, 10)
    m["decay_ic_by_horizon"] = decay_profile(fp, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(fp, lib_sig)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    m["factor"] = name
    rows.append(m)
    print(f"\n=== {name} ===")
    print(f"  IC={m['ic']:.4f}  ICIR={m['icir']:.4f}  hit={m['ic_hit_ratio']:.3f}  n_dates={m['n_ic_dates']}")
    print(f"  coverage_asset_days={m['coverage_asset_days']:.3f}  dates_ge8={m['coverage_dates_ge8']:.3f}  turnover_10d={m['turnover_10d_rank']}")
    print(f"  decay={m['decay_ic_by_horizon']}")
    print(f"  max_lib_corr={corr} ({key})")

print("\n\n=== GATE CHECK (|IC|>=0.0070 and |ICIR|>=0.0840 at h=10) ===")
for r in rows:
    gate = (abs(r["ic"]) >= 0.0070) and (abs(r["icir"]) >= 0.0840)
    print(f"  {r['factor']:18s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f}  {'PASS' if gate else 'fail'}")
