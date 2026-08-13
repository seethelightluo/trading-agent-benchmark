"""miner_1 (2032-11-15): explore batch AE - new cross-asset factor candidates.

Candidates (cross-sectional, 15-asset universe):
  A. dxy_beta_60d       - rolling beta of asset ret on DXY ret (USD macro beta)
  B. vix_beta_60d       - rolling beta of asset ret on VIX change (risk sensitivity)
  C. gold_corr_20d      - rolling corr with XAU ret (safe-haven affinity; XAU self = NaN)
  D. crypto_corr_30d    - rolling corr with BTC ret (risk-asset affinity; BTC self = NaN)
  E. sharpe_60d         - risk-adjusted momentum: mean(ret,60)/std(ret,60)
  F. efficiency_ratio_20d - |ret20| / sum(|ret|,20) trend efficiency
  G. vol_scaled_reversal_10d - -ret10 / vol20 (short reversal)
  H. downside_capture_60d - mean(neg ret)/|mean(pos ret)| over 60d (asymmetry)
  I. range_position_20d - mean((close-low)/(high-low)) over 20d (buying pressure)
  J. cross_z_mom_60d    - cross-sectional z-score of 60d momentum

Admission gates (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Also revalidates the 3 library factors at the same horizon for drift tracking.
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
print(f"closes shape={closes.shape} range={closes.index.min().date()}..{closes.index.max().date()}")

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
dxy_ret = panels["DXY"]["close"].astype(float).pct_change()
vix_chg = panels["VIX"]["close"].astype(float).diff()
xau_ret = rets["XAU"]
btc_ret = rets["BTC"]

cands = {}
# A. USD macro beta 60d
cands["dxy_beta_60d"] = rolling_beta(rets, dxy_ret, 60, 40)
# B. VIX sensitivity 60d (beta on VIX level change)
cands["vix_beta_60d"] = rolling_beta(rets, vix_chg, 60, 40)
# C. gold correlation 20d (XAU self set NaN)
gold_corr = rets.rolling(20, min_periods=15).corr(xau_ret)
gold_corr["XAU"] = np.nan
cands["gold_corr_20d"] = gold_corr
# D. crypto correlation 30d (BTC self set NaN)
crypto_corr = rets.rolling(30, min_periods=20).corr(btc_ret)
crypto_corr["BTC"] = np.nan
cands["crypto_corr_30d"] = crypto_corr
# E. risk-adjusted momentum 60d
cands["sharpe_60d"] = rets.rolling(60, min_periods=40).mean() / rets.rolling(60, min_periods=40).std()
# F. trend efficiency 20d
cands["efficiency_ratio_20d"] = (closes / closes.shift(20) - 1).abs() / rets.abs().rolling(20, min_periods=15).sum()
# G. vol-scaled reversal 10d
cands["vol_scaled_reversal_10d"] = -(closes / closes.shift(10) - 1) / rets.rolling(20).std()
# H. downside capture asymmetry 60d
gains = rets.where(rets > 0)
losses = rets.where(rets < 0)
cands["downside_capture_60d"] = losses.rolling(60, min_periods=40).mean().abs() / gains.rolling(60, min_periods=40).mean()
# I. intraday range position 20d (buying pressure)
hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).sort_index()
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).sort_index()
rng = (closes - lo) / (hi - lo).replace(0, np.nan)
cands["range_position_20d"] = rng.rolling(20, min_periods=15).mean()
# J. cross-sectional z-score of 60d momentum
mom60 = closes / closes.shift(60) - 1
cands["cross_z_mom_60d"] = (mom60 - mom60.mean(axis=1, skipna=True)) / mom60.std(axis=1, skipna=True)

GATE_IC, GATE_ICIR = 0.0070, 0.0840
HORIZON = 10
fwd = forward_returns(closes, HORIZON)

def evaluate(name, panel, expected_sign=1):
    ics = rank_ic_series(panel, fwd, min_valid=8)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(panel, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), 8, expected_sign)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    ics_recent = ics[ics.index >= ics.index.max() - pd.Timedelta(days=365)]
    m["ic_last_250d"] = round(float(ics_recent.mean()), 4) if len(ics_recent) else None
    m["n_ic_dates_recent"] = int(len(ics_recent))
    return m

# library revalidation first (drift tracking)
print("\n===== LIBRARY REVALIDATION (h=10) =====")
lib_results = {}
for name, panel in lib.items():
    m = evaluate(name)
    lib_results[name] = m
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} | recent_1y_IC={m['ic_last_250d']}")

print("\n===== CANDIDATES (h=10) =====")
results = {}
for name, panel in cands.items():
    m = evaluate(name)
    results[name] = m
    passed = abs(m["ic"]) >= GATE_IC and abs(m["icir"]) >= GATE_ICIR
    print(f"\n=== {name} {'PASS' if passed else 'fail'} | IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} | cov_asset={m['coverage_asset_days']:.3f} "
          f"cov_dates>=8={m['coverage_dates_ge8']:.3f} | turn={m['turnover_10d_rank']:.3f} | "
          f"libcorr={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']}) | recent_1y_IC={m['ic_last_250d']}")
    print("   decay:", {k: v for k, v in m["decay_ic_by_horizon"].items()})

print("\n--- summary table ---")
for name, m in results.items():
    passed = abs(m["ic"]) >= GATE_IC and abs(m["icir"]) >= GATE_ICIR
    print(f"{name:28s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} recent1y={m['ic_last_250d']} {'PASS' if passed else '---'}")

# save raw results for audit
out = {"date": "2032-11-15", "batch": "AE", "results": results}
with open("scripts/_miner1_20321115_batchAE_results.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("\nsaved scripts/_miner1_20321115_batchAE_results.json")
