"""miner_1 2026-08-27: persist batch A factors that passed IC/ICIR + crowding gates.

Passing candidates (full-library audit, canonical grid 2020-01-02..2026-07-15):
  - comm_basket_beta_60   IC=0.0428 ICIR=0.1219 max_full_lib_rho=0.181
  - down_beta_hs300_60    IC=-0.0379 ICIR=-0.1104 max_full_lib_rho=0.080
Rejected: mom_x_er_20 (rho=0.967 vs spx_rel_mom_20) - crowded.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           persist_factor, canonical_grid, build_library_panels,
                           max_library_correlation)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"grid: {len(grid)} dates {grid.min().date()}..{grid.max().date()}")

hs300_r = prices['000300.SH']['close'].pct_change().rename('hs300')

def ew_ret(symbols, prices):
    df = None
    for s in symbols:
        r = prices[s]['close'].pct_change().rename(s)
        df = r if df is None else pd.concat([df, r], axis=1)
    return df.mean(axis=1).rename('ew')

comm_r = ew_ret(['XAU', 'COPPER', 'WTI'], prices)

def f_comm_basket_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), comm_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_down_beta_hs300_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), hs300_r.rename('y')], axis=1).dropna()
    z = z[z['y'] < 0]
    if len(z) < 30:
        return pd.Series(np.nan, index=df.index)
    return (z['r'].rolling(60, min_periods=20).cov(z['y']) / z['y'].rolling(60, min_periods=20).var()).reindex(df.index)

lib = build_library_panels(prices)

specs = [
    dict(
        factor_id='comm_basket_beta_60',
        factor_name='Commodity basket beta (60d)',
        expression='rolling_beta(asset_daily_ret, ew_ret(XAU,COPPER,WTI), 60)',
        description='60-day rolling regression beta of each asset daily return on the equal-weight '
                    'commodity basket (XAU/COPPER/WTI) return. Assets positively exposed to the broad '
                    'commodity complex tend to outperform; acts as an inflation/real-asset regime tilt.',
        dependencies=['close', 'XAU close', 'COPPER close', 'WTI close'],
        parameters={'window': 60, 'basket': ['XAU', 'COPPER', 'WTI'], 'weighting': 'equal', 'admission_horizon': 10},
        expected_direction=1,
        tags=['beta', 'macro', 'commodity', 'cross-asset', 'regime'],
        regime_notes='2020-2026 multi-regime incl. 2020 recovery, 2021-22 commodity supercycle, 2023-24 '
                     'range, 2025-26 supply shocks. Positive IC strongest at 10-20d horizon; turnover low '
                     '(1.48 rank units/10d) making it cheap to trade.',
        version='1.0.0',
    ),
    dict(
        factor_id='down_beta_hs300_60',
        factor_name='Downside beta vs China equities (60d)',
        expression='rolling_beta(asset_daily_ret | hs300_ret<0, hs300_daily_ret | hs300_ret<0, 60)',
        description='60-day rolling beta of each asset daily return on CSI300 (000300.SH) return, estimated '
                    'only on days when the CSI300 itself fell. Assets with high downside linkage to Chinese '
                    'equities tend to underperform (risk-off contagion); expected_direction=-1.',
        dependencies=['close', '000300.SH close'],
        parameters={'window': 60, 'min_periods': 20, 'condition': 'hs300_ret<0', 'min_obs': 30, 'admission_horizon': 10},
        expected_direction=-1,
        tags=['beta', 'downside', 'china', 'risk-off', 'cross-asset'],
        regime_notes='2020-2026 multi-regime. Negative IC stable at 10d; note moderate coverage '
                     '(cov=0.324, 761 IC dates with >=8 instruments) because estimation requires '
                     'downside days; complementary to existing down_beta_60 (SPX-based) with low '
                     'library correlation (rho=0.080).',
        version='1.0.0',
    ),
]

for sp in specs:
    fn = f_comm_basket_beta_60 if sp['factor_id'] == 'comm_basket_beta_60' else f_down_beta_hs300_60
    panel = factor_to_panel(fn, prices)
    m = validate_factor(sp['factor_id'], panel, prices)
    rho, rho_id = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f"{sp['factor_id']}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} rho={rho:.3f}({rho_id}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print('  SKIP persistence (gate failed)')
        continue
    path, arr = persist_factor(
        factor_id=sp['factor_id'],
        factor_name=sp['factor_name'],
        expression=sp['expression'],
        description=sp['description'],
        dependencies=sp['dependencies'],
        parameters=sp['parameters'],
        expected_direction=sp['expected_direction'],
        panel=panel,
        metrics=m,
        tags=sp['tags'],
        grid=grid,
        prices=prices,
        version=sp['version'],
        status='EFFECTIVE',
        regime_notes=sp['regime_notes'],
    )
    print(f"  persisted -> {path} (artifact shape {arr.shape})")

print('DONE')
