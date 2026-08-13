"""miner_3 (2032-10-18): explore batch AD - novel cross-asset factor candidates.

Candidates (all cross-sectional, 15-asset universe):
  A. skew_20d            - rolling 20d return skewness (crash-risk)
  B. efficiency_ratio_60d - trend efficiency |ret60| / sum(|ret|,60)
  C. ret_autocorr_20d    - 1st-order return autocorrelation over 20d
  D. max_ret_20d         - MAX lottery effect: max daily return over 20d
  E. us10y_beta_60d      - rolling beta of asset ret on US10Y yield change
  F. downside_vol_ratio_20d - downside deviation / total vol over 20d
  G. gain_loss_asym_60d  - mean(gain)/|mean(loss)| over 60d
  H. vol_adj_reversal_5d - -5d return / 20d vol (scaled short reversal)

Admission gates (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, max_library_corr)

panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()

# ---------------- library signals (3 effective factors) ----------------
def rolling_beta(a, b, win, min_obs):
    cov = a.rolling(win, min_periods=min_obs).cov(b)
    var = b.rolling(win, min_periods=min_obs).var()
    return cov / var

mkt = rets.mean(axis=1, skipna=True)
dn_mask = (mkt < 0).astype(float)
a_dn = rets.where(dn_mask > 0)
m_dn = mkt.where(dn_mask > 0)
lib = {}
lib["vol_adj_mom_accel_20x60"] = ((closes / closes.shift(20) - 1) - (closes / closes.shift(60) - 1)) / rets.rolling(20).std()
lib["dn_mkt_beta_60d"] = a_dn.rolling(60, min_periods=40).cov(m_dn) / m_dn.rolling(60, min_periods=40).var()
cn10y_ret = panels["CN10Y"]["close"].astype(float).pct_change()
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)

# ---------------- candidate factors ----------------
cands = {}
# A. skewness 20d
cands["skew_20d"] = rets.rolling(20).skew()
# B. efficiency ratio 60d
cands["efficiency_ratio_60d"] = (closes / closes.shift(60) - 1).abs() / rets.abs().rolling(60).sum()
# C. return autocorr 20d
# vectorized lag-1 autocorr over 20d: cov(ret_t, ret_{t-1})/var(ret_t)
_rets_lag = rets.shift(1)
_cov = (rets * _rets_lag).rolling(20).mean() - rets.rolling(20).mean() * _rets_lag.rolling(20).mean()
cands["ret_autocorr_20d"] = _cov / rets.rolling(20).var()
# D. MAX lottery 20d
cands["max_ret_20d"] = rets.rolling(20).max()
# E. US10Y beta 60d
us10y_ret = panels["US10Y"]["close"].astype(float).pct_change()
cands["us10y_beta_60d"] = rolling_beta(rets, us10y_ret, 60, 40)
# F. downside vol ratio 20d
downside = rets.where(rets < 0)
dd = np.sqrt((downside ** 2).rolling(20).mean())
cands["downside_vol_ratio_20d"] = dd / rets.rolling(20).std()
# G. gain/loss asymmetry 60d
gains = rets.where(rets > 0)
losses = rets.where(rets < 0)
cands["gain_loss_asym_60d"] = gains.rolling(60).mean() / losses.rolling(60).mean().abs()
# H. vol-adj reversal 5d
cands["vol_adj_reversal_5d"] = -(closes / closes.shift(5) - 1) / rets.rolling(20).std()

GATE_IC, GATE_ICIR = 0.0070, 0.0840
HORIZON = 10

fwd = forward_returns(closes, HORIZON)
results = {}
for name, panel in cands.items():
    ics = rank_ic_series(panel, fwd, min_valid=8)
    m = summarize_ic(ics, expected_sign=1)
    m.update(coverage_metrics(panel, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    # recent drift: last 250 trading days
    ics_recent = ics[ics.index >= ics.index.max() - pd.Timedelta(days=365)]
    m["ic_last_250d"] = round(float(ics_recent.mean()), 4) if len(ics_recent) else None
    m["n_ic_dates_recent"] = int(len(ics_recent))
    results[name] = m
    passed = abs(m["ic"]) >= GATE_IC and abs(m["icir"]) >= GATE_ICIR
    print(f"\n=== {name} {'PASS' if passed else 'fail'} | IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} | cov_asset={m['coverage_asset_days']:.3f} "
          f"cov_dates>=8={m['coverage_dates_ge8']:.3f} | turn={m['turnover_10d_rank']:.3f} | "
          f"libcorr={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']}) | recent_1y_IC={m['ic_last_250d']}")
    print("   decay:", {k: v for k, v in m["decay_ic_by_horizon"].items()})

print("\n--- summary table ---")
for name, m in results.items():
    passed = abs(m["ic"]) >= GATE_IC and abs(m["icir"]) >= GATE_ICIR
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} {'PASS' if passed else '---'}")
