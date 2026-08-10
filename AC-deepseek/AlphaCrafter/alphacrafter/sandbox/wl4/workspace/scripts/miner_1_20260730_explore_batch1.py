"""Batch-1 factor exploration for miner_1 (2026-07-30).

Screens ~10 candidate factor families on the 15-asset cross-asset universe.
Admission gates (h=10): |IC|>=0.007, |ICIR|>=0.084.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, ret_panel, library_signals,
                                 full_eval, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
vix = panels["VIX"]["close"].astype(float)
library = library_signals(panels, closes, rets, vix)

print(f"Panel: {closes.shape[0]} dates, {closes.shape[1]} assets; range "
      f"{closes.index[0].date()}..{closes.index[-1].date()}")

def add(p, name, exp_sign=1):
    p = p.reindex(closes.index)
    m, ics = full_eval(p, closes, library=library, expected_sign=exp_sign)
    print(f"\n=== {name} (dir={exp_sign}) ===")
    print(f"  IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} ndates={m['n_ic_dates']}")
    print(f"  cov_asset={m['coverage_asset_days']} cov_dates_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']}")
    print(f"  decay={m['decay_ic_by_horizon']}")
    print(f"  max_lib_corr={m['max_abs_library_correlation']} (vs {m.get('max_corr_factor')})")
    return m

# ---- Candidate constructions ----
cands = {}

# 1. Range position 20d: where close sits inside 20d high-low range
hi20 = pd.concat({a: panels[a]['high'].astype(float) for a in TRADABLE}, axis=1).rolling(20).max()
lo20 = pd.concat({a: panels[a]['low'].astype(float) for a in TRADABLE}, axis=1).rolling(20).min()
cands['range_pos_20d'] = (closes - lo20) / (hi20 - lo20).replace(0, np.nan)

# 2. Skewness of 20d returns
cands['skew_20d'] = rets.rolling(20).skew()

# 3. Downside risk ratio: std of negative returns / std of returns
neg = rets.clip(upper=0)
cands['downside_ratio_20d'] = neg.rolling(20).std() / rets.rolling(20).std()

# 4. Amihud illiquidity 20d (log) - volume may be zero for yield series
vol = pd.concat({a: panels[a]['volume'].astype(float) for a in TRADABLE}, axis=1)
illiq = (rets.abs() / vol.replace(0, np.nan)).rolling(20).mean()
cands['amihud_illiq_20d'] = np.log1p(illiq * 1e9)

# 5. Risk-adjusted trend: 60d mean / 20d std (Sharpe-like)
cands['sharpe_trend_60x20'] = rets.rolling(60).mean() / rets.rolling(20).std()

# 6. Distance from 60d high
hi60 = pd.concat({a: panels[a]['high'].astype(float) for a in TRADABLE}, axis=1).rolling(60).max()
cands['dist_high_60d'] = closes / hi60 - 1.0

# 7. DXY beta 60d: regression beta of asset returns on DXY returns
dxy = panels['DXY']['close'].astype(float).pct_change()
dxy_beta = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename('a'), dxy.rename('d')], axis=1).dropna()
    b = z['a'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
    dxy_beta[a] = b
cands['dxy_beta_60d'] = pd.DataFrame(dxy_beta, index=rets.index)

# 8. Kaufman efficiency ratio 20d
cands['eff_ratio_20d'] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()

# 9. Short-term reversal 5d
cands['reversal_5d'] = -(closes / closes.shift(5) - 1.0)

# 10. Vol term structure: 20d vol / 60d vol
cands['vol_ratio_20x60'] = rets.rolling(20).std() / rets.rolling(60).std()

# 11. VIX-change sensitivity (unconditional beta magnitude)
vix_ret = vix.pct_change()
vix_beta = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename('a'), vix_ret.rename('v')], axis=1).dropna()
    b = z['a'].rolling(40).cov(z['v']) / z['v'].rolling(40).var()
    vix_beta[a] = b
cands['vix_beta_40d'] = pd.DataFrame(vix_beta, index=rets.index)

# 12. Cross-asset dispersion / correlation with equal-weight market (risk-on beta)
mkt = rets.mean(axis=1)
mkt_beta = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename('a'), mkt.rename('m')], axis=1).dropna()
    b = z['a'].rolling(60).cov(z['m']) / z['m'].rolling(60).var()
    mkt_beta[a] = b
cands['market_beta_60d'] = pd.DataFrame(mkt_beta, index=rets.index)

print("\n" + "=" * 80)
for name, p in cands.items():
    sign = -1 if name in ("reversal_5d",) else 1
    add(p, name, exp_sign=sign)
