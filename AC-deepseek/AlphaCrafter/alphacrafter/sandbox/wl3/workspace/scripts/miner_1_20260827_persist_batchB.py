"""miner_1 2026-08-27: persist batch B factors (adx_14, mom_skew_change, hilo_vol_ratio_20).

All passed |IC|>=0.007 & |ICIR|>=0.084 at 10d horizon and full-library artifact
crowding audit (max rho < 0.5; observed max <= 0.094).
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

def f_adx_14(df, s):
    up = df['high'].diff()
    dn = -df['low'].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(14).mean()

def f_mom_skew_change(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).skew() - r.rolling(20).skew()

def f_hilo_vol_ratio_20(df, s):
    c = df['close']
    rng = (c.rolling(20).max() - c.rolling(20).min()) / c
    vol = c.pct_change().rolling(20).std()
    return rng / vol

fns = {'adx_14': f_adx_14, 'mom_skew_change': f_mom_skew_change, 'hilo_vol_ratio_20': f_hilo_vol_ratio_20}

specs = [
    dict(
        factor_id='adx_14',
        factor_name='ADX trend strength (14d)',
        expression='ADX = 14d_sma(100 * |DI+ - DI-| / (DI+ + DI-))',
        description='Wilder ADX: smoothed directional movement index measuring trend strength '
                    'regardless of direction. Assets in strong trends (high ADX) tend to outperform '
                    'over 10d horizon; trend-persistence premium across cross-asset universe.',
        dependencies=['high', 'low', 'close'],
        parameters={'period': 14, 'smoothing': 14, 'admission_horizon': 10},
        expected_direction=1,
        tags=['trend', 'momentum', 'technical', 'persistence'],
        regime_notes='2020-2026 multi-regime. Positive IC at 10d (IC=0.034, ICIR=0.110); decay '
                     'monotonic increasing through 20d. Low crowding vs library (max rho=0.094).',
        version='1.0.0',
    ),
    dict(
        factor_id='mom_skew_change',
        factor_name='Momentum of skewness (60d-20d)',
        expression='skew(ret,60) - skew(ret,20)',
        description='Change in return skewness: short-window skew minus long-window skew. Assets whose '
                    'recent return distribution has become more left-skewed relative to their longer '
                    'history (falling factor) tend to outperform - rising negative skew flags '
                    'deteriorating risk. expected_direction=-1.',
        dependencies=['close'],
        parameters={'short_window': 20, 'long_window': 60, 'admission_horizon': 10},
        expected_direction=-1,
        tags=['skew', 'tail-risk', 'distributional', 'technical'],
        regime_notes='2020-2026 multi-regime. Negative IC at 10d (IC=-0.028, ICIR=-0.089); stable '
                     'across horizons; low crowding (max rho=0.084).',
        version='1.0.0',
    ),
    dict(
        factor_id='hilo_vol_ratio_20',
        factor_name='20d range-to-volatility ratio',
        expression='(rolling_max(close,20)-rolling_min(close,20))/close / std(ret,20)',
        description='Total 20d high-low range normalized by close and by 20d realized volatility. '
                    'High values indicate the range is not explained by day-to-day volatility '
                    '(persistent directional drift / gap moves); assets with high range efficiency '
                    'tend to outperform over 10d.',
        dependencies=['close'],
        parameters={'window': 20, 'admission_horizon': 10},
        expected_direction=1,
        tags=['range', 'volatility', 'technical', 'trend-quality'],
        regime_notes='2020-2026 multi-regime. Positive IC at 10d (IC=0.042, ICIR=0.129), strongest '
                     'of batch B; higher turnover (4.0 rank units/10d) priced in at rebalance '
                     'frequency. Low crowding (max rho=0.074).',
        version='1.0.0',
    ),
]

lib = build_library_panels(prices)
for sp in specs:
    fid = sp['factor_id']
    panel = factor_to_panel(fns[fid], prices)
    m = validate_factor(fid, panel, prices)
    rho, rho_id = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f"{fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} rho={rho:.3f}({rho_id}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        continue
    path, arr = persist_factor(
        factor_id=fid, factor_name=sp['factor_name'], expression=sp['expression'],
        description=sp['description'], dependencies=sp['dependencies'],
        parameters=sp['parameters'], expected_direction=sp['expected_direction'],
        panel=panel, metrics=m, tags=sp['tags'], grid=grid, prices=prices,
        version=sp['version'], status='EFFECTIVE', regime_notes=sp['regime_notes'],
    )
    print(f"  persisted -> {path} (artifact {arr.shape})")

print('DONE')
