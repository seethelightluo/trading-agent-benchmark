"""miner_1 exploration 2029-06-18: candidate factor screen on 15-asset universe.

Window: <= 2029-06-17 (previous completed trading day). Anti-leakage enforced.
Candidates (motivation: VIX-62 stress tape, defensive rotation):
  - drawdown_60d      : distance from 60d high (relative strength / dip)
  - vix_beta_60d      : beta on VIX changes (risk-on/off sensitivity)
  - dxy_beta_60d      : beta on DXY changes (USD sensitivity)
  - skew_20d          : rolling return skewness (crash-risk)
  - ma_trend_60       : close / MA60 - 1 (trend following)
  - sharpe_60d        : risk-adjusted 60d momentum
  - ret_5d            : raw 5d return (reversal/momentum probe, sign from IC)
  - asym_vol_20       : downside/upside vol ratio (asymmetric tail risk)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from miner1_iclib import (load_prices, load_macro, rank_ic_series, summarize_ic,
                          library_corr, rolling_beta_fast)

px = load_prices()
rets = px.pct_change()
macro = load_macro()
print('price panel:', px.shape, px.index.min().date(), '->', px.index.max().date())
print('assets:', list(px.columns))

# ---- library factor values (recomputed from prices, for correlation reference) ----
mkt = px.mean(axis=1)
dn_x = mkt.clip(upper=0.0)
lib = {}
lib['dn_mkt_beta_60d'] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), dn_x, 60, 40))
cn10y_ret = px['CN10Y'].pct_change()
lib['rate_beta_cn10y_60d'] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), cn10y_ret, 60, 40))

def vol_adj_mom_accel(c, fast=20, slow=60, vol_win=20):
    r = c.pct_change()
    return (c / c.shift(fast) - 1.0 - (c / c.shift(slow) - 1.0)) / r.rolling(vol_win).std()
lib['vol_adj_mom_accel_20x60'] = px.apply(vol_adj_mom_accel)

# ---- candidates ----
cand = {}
cand['drawdown_60d'] = px / px.rolling(60, min_periods=40).max() - 1.0
cand['ma_trend_60'] = px / px.rolling(60, min_periods=40).mean() - 1.0
cand['skew_20d'] = rets.rolling(20, min_periods=15).skew()
cand['sharpe_60d'] = rets.rolling(60, min_periods=40).mean() / rets.rolling(60, min_periods=40).std()
cand['ret_5d'] = px / px.shift(5) - 1.0
pos_r = rets.where(rets > 0, 0.0)
neg_r = rets.where(rets < 0, 0.0)
cand['asym_vol_20'] = neg_r.rolling(20, min_periods=15).std() / pos_r.rolling(20, min_periods=15).std().replace(0, np.nan)

vix_ret = macro['VIX'].pct_change()
dxy_ret = macro['DXY'].pct_change()
cand['vix_beta_60d'] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), vix_ret, 60, 40))
cand['dxy_beta_60d'] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), dxy_ret, 60, 40))

# ---- forward returns at multiple horizons ----
fwd = {}
for h in [1, 2, 3, 5, 10, 20]:
    fwd[h] = px.shift(-h) / px - 1.0

print('\n=== candidate screen (h=10 IC / ICIR) ===')
for name, fv in cand.items():
    ic10 = rank_ic_series(fv, fwd[10])
    s = summarize_ic(ic10, name=name, fval=fv, fwd_map=fwd)
    mcorr, mfid, _ = library_corr(fv, lib)
    s['max_abs_library_correlation'] = round(mcorr, 4)
    s['max_corr_factor'] = mfid
    gate_ic = abs(s.get('ic', 0)) >= 0.007
    gate_icir = abs(s.get('icir', 0)) >= 0.084
    s['GATE'] = 'PASS' if (gate_ic and gate_icir) else 'fail'
    print(json.dumps(s, default=str))

# also report library factor current IC for re-validation reference
print('\n=== library factor recent IC (full window) ===')
for name, fv in lib.items():
    ic10 = rank_ic_series(fv, fwd[10])
    s = summarize_ic(ic10, name=name, fval=fv, fwd_map=fwd)
    print(json.dumps({k: s.get(k) for k in ['name', 'n_ic_dates', 'ic', 'icir', 'ic_hit_ratio', 'coverage_asset_days']}, default=str))

# regime snapshot
print('\n=== regime snapshot ===')
mkt_ret = rets.mean(axis=1)
for w in [20, 60]:
    r = (1 + mkt_ret).rolling(w).apply(np.prod, raw=True) - 1
    v = mkt_ret.rolling(w).std() * np.sqrt(252)
    print(f'mkt(live) {w:3d}d cum: {r.iloc[-1]*100:+.2f}%  vol_ann: {v.iloc[-1]*100:.1f}%')
print('VIX last:', round(macro['VIX'].iloc[-1], 2), ' 60d ago:', round(macro['VIX'].iloc[-61], 2))
print('DXY last:', round(macro['DXY'].iloc[-1], 2), ' 60d ago:', round(macro['DXY'].iloc[-61], 2))
