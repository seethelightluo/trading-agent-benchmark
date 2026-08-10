"""miner_1 broad candidate screen AFTER quarantine cycle (worldline_pairwise_signal_quality_v1).

Goal: find NEW factors passing |IC|>=0.007 and |ICIR|>=0.084 at horizon 10,
with recoverable signal artifacts, low correlation to the 4 former library
factors (mom_10d_skip5, mom_120d_skip5, vol_of_vol20x60, vix_beta_cond_60x20),
and regime robustness across 2020-2022 / 2023-2026.
"""
import sys, math
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
from factor_validation_lib import (TRADABLE, MIN_INSTR, load_panel, load_macro,
                                   align_fwd_returns, rank_ic_series, ic_analysis,
                                   print_report, library_corr)

VISIBLE = "2026-07-29"
panel = load_panel(max_date=VISIBLE)
ret = panel.pct_change()
vix = load_macro("VIX", max_date=VISIBLE)
vixr = vix.pct_change()
print(f"panel: {panel.shape}  assets={panel.shape[1]}  dates={panel.shape[0]}  "
      f"through {panel.index.max().date()}")

vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
vol90 = ret.rolling(90).std()

def beta_of(a, m, win):
    return a.rolling(win).cov(m) / m.rolling(win).var()

C = {}
# --- momentum family (new variants) ---
for lb in (60, 90, 250):
    C[f'mom_{lb}d_skip5'] = panel.shift(5) / panel.shift(5 + lb) - 1.0
C['mom_20d_skip5_risk_adj'] = (panel.shift(5) / panel.shift(25) - 1.0) / vol20
C['mom_60d_skip5_risk_adj'] = (panel.shift(5) / panel.shift(65) - 1.0) / vol60
C['trend_20d'] = panel / panel.rolling(20).mean() - 1.0
C['trend_60d'] = panel / panel.rolling(60).mean() - 1.0
C['price_vs_52w_high'] = panel / panel.rolling(250).max() - 1.0
C['dd_from_60d_high'] = panel / panel.rolling(60).max() - 1.0
C['curvature_20x60'] = (panel / panel.rolling(20).mean()) / (panel / panel.rolling(60).mean()) - 1.0

# --- volatility family ---
C['inv_vol_20d'] = -vol20
C['inv_vol_60d'] = -vol60
C['vol_ratio_20x60'] = vol20 / vol60
C['vol_ratio_10x60'] = ret.rolling(10).std() / vol60
C['downside_vol_20d'] = -ret.clip(upper=0).rolling(20).std()
C['downside_vol_ratio_20x60'] = ret.clip(upper=0).rolling(20).std() / ret.clip(upper=0).rolling(60).std()
C['skew_60d'] = ret.rolling(60).skew()
C['kurt_60d'] = ret.rolling(60).kurt()
C['realized_vol_z_20'] = (vol20 - vol20.rolling(120).mean()) / vol20.rolling(120).std()
C['range_20d'] = (panel.rolling(20).max() - panel.rolling(20).min()) / panel.rolling(20).mean()

# --- macro-conditional / beta ---
C['dxy_beta_60d'] = beta_of(ret, load_macro('DXY', max_date=VISIBLE).pct_change(), 60)
C['vix_beta_60d'] = beta_of(ret, vixr, 60)
C['vix_beta_neg_60d'] = -beta_of(ret, vixr, 60)
C['usdcny_beta_60d'] = beta_of(ret, load_macro('USDCNY', max_date=VISIBLE).pct_change(), 60)
C['vol_state_mom20'] = (panel.shift(5) / panel.shift(25) - 1.0) * (vol20 < vol20.rolling(120).median())
C['mom60_lowvol_cond'] = (panel.shift(5) / panel.shift(65) - 1.0) * (vol20.rank(axis=1, pct=True) < 0.5)

# --- liquidity / volume ---
vol_series = {s: get_stock_daily_data(symbol=s, days=4000) for s in TRADABLE}
vol_panel = pd.DataFrame({s: df.set_index(pd.to_datetime(df['date']))['volume'].astype(float)
                          for s, df in vol_series.items() if df is not None}).sort_index()
vol_panel = vol_panel[vol_panel.index <= pd.Timestamp(VISIBLE)]
C['amihud_20d'] = (ret.abs() / panel).rolling(20).mean()
C['volume_z_20d'] = (vol_panel - vol_panel.rolling(60).mean()) / vol_panel.rolling(60).std()
C['volume_trend_10x60'] = vol_panel.rolling(10).mean() / vol_panel.rolling(60).mean() - 1.0

# --- cross-asset: yield-related ---
C['us10y_mom_20d'] = panel['US10Y'].pct_change(20).reindex(panel.index)  # single asset -> broadcast
# better: yield-curve term signal as asset-specific own-rate momentum
C['yield_mom_20d'] = panel.pct_change(20)

# --- library signals (former effective) for correlation audit ---
lib = {}
lib['mom_10d_skip5'] = panel.shift(5) / panel.shift(15) - 1.0
lib['mom_120d_skip5'] = panel.shift(5) / panel.shift(125) - 1.0
lib['vol_of_vol20x60'] = vol20.rolling(60).std()
lib['vix_beta_cond_60x20'] = -beta_of(ret, vixr, 60) * (vix / vix.shift(20) - 1.0)

print('=' * 110)
results = {}
for name, f in C.items():
    res = ic_analysis(f, panel, horizon=10, label=name)
    results[name] = res
    # subperiod IC at horizon 10
    ic10 = rank_ic_series(f, align_fwd_returns(panel, 10)).dropna()
    sub1 = ic10[(ic10.index >= '2020-01-01') & (ic10.index <= '2022-12-31')]
    sub2 = ic10[(ic10.index >= '2023-01-01')]
    s1 = f'{sub1.mean():.4f}(n={len(sub1)})' if len(sub1) else 'na'
    s2 = f'{sub2.mean():.4f}(n={len(sub2)})' if len(sub2) else 'na'
    rho = library_corr(f, lib)
    ok = (abs(res['ic'] or 0) >= 0.007) and (abs(res['icir'] or 0) >= 0.084)
    print(f"{'PASS' if ok else '----'} {name:<28} ic={res['ic']} icir={res['icir']} "
          f"hit={res['ic_hit_ratio']} turn={res['turnover_10d_rank']} cov={res['coverage_asset_days']} "
          f"librho={rho:.3f} | sub20-22={s1} sub23-26={s2}")
print('=' * 110)

# dump compact results to json for persistence step
import json
summary = {k: {kk: vv for kk, vv in v.items() if kk != 'decay_ic_by_horizon'} for k, v in results.items()}
json.dump(summary, open('scripts/_screen_v2_results.json', 'w'), indent=1)
print('saved scripts/_screen_v2_results.json')
