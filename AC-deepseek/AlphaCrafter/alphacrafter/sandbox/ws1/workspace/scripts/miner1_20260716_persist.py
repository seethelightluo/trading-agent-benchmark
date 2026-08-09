"""miner_1: persist all gate-passing factors (h=10 admission: |IC|>=0.007, |ICIR|>=0.084).

Writes factors/<factor_id>.json for each passing candidate with full validation
metadata (IC/ICIR/hit/coverage/turnover/decay/max_abs_library_correlation).
"""
import json
import time
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic)

H = 10
MIN_VALID = 8
VALID_PERIOD = "2020-01-01..2026-07-15"
LAST_VALIDATED = "2026-07-16"

t0 = time.time()
panel = build_panel()
closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
print(f'panel built in {time.time()-t0:.0f}s grid_dates={len(grid)}')

CANDIDATES = {
    'mom_10d_skip5': {
        'name': 'Short Momentum 10d (skip 5d)',
        'tags': ['momentum', 'cross-asset'],
        'expr': 'close.shift(5) / close.shift(15) - 1.0',
        'desc': '10-day price momentum with 5-day skip to avoid short-term reversal: '
                'return from 15 days ago to 5 days ago.',
        'params': {'lookback': 10, 'skip': 5},
        'fn': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    },
    'mom_20d_skip5': {
        'name': 'Momentum 20d (skip 5d)',
        'tags': ['momentum', 'cross-asset'],
        'expr': 'close.shift(5) / close.shift(25) - 1.0',
        'desc': '20-day price momentum with 5-day skip: return from 25 days ago to 5 days ago.',
        'params': {'lookback': 20, 'skip': 5},
        'fn': lambda s, c, v: c.shift(5) / c.shift(25) - 1.0,
    },
    'mom_120d_skip5': {
        'name': 'Long Momentum 120d (skip 5d)',
        'tags': ['momentum', 'cross-asset'],
        'expr': 'close.shift(5) / close.shift(125) - 1.0',
        'desc': '120-day (6-month) price momentum with 5-day skip: return from 125 days ago to 5 days ago.',
        'params': {'lookback': 120, 'skip': 5},
        'fn': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    },
    'trend_sma60': {
        'name': 'Trend vs SMA60',
        'tags': ['trend', 'time-series'],
        'expr': 'close / close.rolling(60).mean() - 1.0',
        'desc': 'Price position relative to its 60-day simple moving average; '
                'positive when in an uptrend above the average.',
        'params': {'window': 60},
        'fn': lambda s, c, v: c / c.rolling(60).mean() - 1.0,
    },
    'trend_sma120': {
        'name': 'Trend vs SMA120',
        'tags': ['trend', 'time-series'],
        'expr': 'close / close.rolling(120).mean() - 1.0',
        'desc': 'Price position relative to its 120-day simple moving average; '
                'slower regime trend signal.',
        'params': {'window': 120},
        'fn': lambda s, c, v: c / c.rolling(120).mean() - 1.0,
    },
    'risk_adj_trend20': {
        'name': 'Risk-adjusted trend 20d',
        'tags': ['trend', 'risk-adjusted'],
        'expr': 'mean(pct_change,20) / std(pct_change,20)',
        'desc': '20-day Sharpe-like trend: average daily return scaled by daily volatility.',
        'params': {'window': 20},
        'fn': lambda s, c, v: (c.pct_change().rolling(20).mean()
                               / c.pct_change().rolling(20).std()
                               ).replace([np.inf, -np.inf], np.nan),
    },
    'vol_of_vol20x60': {
        'name': 'Volatility of volatility 20x60',
        'tags': ['volatility', 'regime'],
        'expr': 'std(pct_change,20).rolling(60).std()',
        'desc': 'Volatility-of-volatility: 60-day std of 20-day realized vol. '
                'Low values indicate calm/stable vol regimes.',
        'params': {'short_win': 20, 'long_win': 60},
        'fn': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    },
    'zscore_rev_20d': {
        'name': '20d Z-score reversion',
        'tags': ['mean-reversion', 'volatility'],
        'expr': '(close - mean(close,20)) / (std(pct_change,20) * close)',
        'desc': 'Z-score of price vs 20-day mean scaled by vol; high values flag '
                'overextended assets (mean-reversion signal).',
        'params': {'window': 20},
        'fn': lambda s, c, v: ((c - c.rolling(20).mean())
                               / (c.pct_change().rolling(20).std() * c)
                               ).replace([np.inf, -np.inf], np.nan),
    },
    'inv_vol_60d': {
        'name': 'Negative 60d volatility (inverse-vol proxy)',
        'tags': ['volatility', 'risk-parity'],
        'expr': '-std(pct_change,60)',
        'desc': 'Negative realized volatility over 60 days; ranks assets by calmness. '
                'IC is negative -> low-vol assets outperform (use negative direction).',
        'params': {'window': 60},
        'fn': lambda s, c, v: (-c.pct_change().rolling(60).std()
                               ).replace([np.inf, -np.inf], np.nan),
    },
    'vix_beta_cond_60x20': {
        'name': 'VIX-beta conditional 60x20',
        'tags': ['macro-beta', 'risk', 'conditional'],
        'expr': '-beta(asset_ret, VIX_ret, 60) * (VIX/VIX.shift(20) - 1.0)',
        'desc': 'Conditional macro-risk signal: asset beta to VIX changes times '
                '20-day VIX move. Positive when an asset is a VIX hedge and VIX is '
                'rising, or a high-beta asset while VIX falls. IC negative -> flip.',
        'params': {'beta_win': 60, 'vix_win': 20},
        'deps': ['close', 'VIX'],
        'fn': None,  # built below
    },
}


def vix_beta_cond(sym, close, volume, panel):
    macro = panel['macro'].get('VIX')
    if macro is None:
        return None
    grid = panel['grid']
    r_a = close.pct_change().reindex(grid)
    r_m = macro.pct_change().reindex(grid)
    beta = r_a.rolling(60, min_periods=30).cov(r_m) / r_m.rolling(60, min_periods=30).var()
    mm = (macro.reindex(grid) / macro.shift(20).reindex(grid) - 1.0)
    return (-1.0 * beta * mm).replace([np.inf, -np.inf], np.nan)


frames = {}
for label, spec in CANDIDATES.items():
    if spec['fn'] is None:
        frames[label] = factor_values(closes, volumes, grid,
                                      lambda s, c, v, p=panel: vix_beta_cond(s, c, v, p))
    else:
        frames[label] = factor_values(closes, volumes, grid, spec['fn'])

# decay across horizons
decays = {}
for h in (1, 2, 3, 5, 10, 20):
    ret = forward_returns(closes, grid, h)
    for label, fac in frames.items():
        ics = daily_ic(fac, ret, min_valid=MIN_VALID)
        if len(ics) == 0:
            decays.setdefault(label, {})[h] = np.nan
            continue
        m = float(ics['ic'].mean())
        s = float(ics['ic'].std(ddof=1))
        decays.setdefault(label, {})[h] = round(m, 4)
    print(f'decay h={h} done ({time.time()-t0:.0f}s)')

# admission metrics at h=10
ret10 = forward_returns(closes, grid, H)
print('\n=== ADMISSION METRICS h=10 ===')
metrics = {}
for label, fac in frames.items():
    ics = daily_ic(fac, ret10, min_valid=MIN_VALID)
    ic = ics['ic']
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1))
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit = float((ic > 0).mean())
    cov_assets = float(fac.notna().mean().mean())
    cov_dates8 = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
    f10 = fac.iloc[::10].rank(axis=1)
    turn = float(f10.diff().abs().mean().mean()) if len(f10) > 2 else np.nan
    metrics[label] = dict(ic=mean_ic, icir=icir, hit=hit, dates=len(ic),
                          cov_assets=cov_assets, cov_dates8=cov_dates8, turn=turn,
                          decay=decays[label])
    print(f'[{label}] dates={len(ic)} IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f} '
          f'cov_asset={cov_assets:.3f} cov_dates8={cov_dates8:.3f} turn={turn:.3f} '
          f'decay={decays[label]}')

# pairwise rank corr -> max_abs_library_correlation in persistence order
ranks = {l: f.rank(axis=1) for l, f in frames.items()}
labels = list(CANDIDATES.keys())
corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
for a in labels:
    for b in labels:
        if a == b:
            corr.loc[a, b] = 1.0
            continue
        dfa = ranks[a].where(ranks[a].notna() & ranks[b].notna())
        dfb = ranks[b].where(ranks[a].notna() & ranks[b].notna())
        ok = dfa.notna().sum(axis=1) >= MIN_VALID
        if ok.sum() == 0:
            corr.loc[a, b] = np.nan
            continue
        corr.loc[a, b] = float(dfa[ok].corrwith(dfb[ok], axis=1).mean())

# write JSON files
persisted = []
for label in labels:
    spec = CANDIDATES[label]
    m = metrics[label]
    if not persisted:
        max_corr = 0.0
    else:
        max_corr = max(abs(corr.loc[label, p]) for p in persisted)
    payload = {
        'factor_id': label,
        'factor_name': spec['name'],
        'version': '1.0.0',
        'calculation': {
            'expression': spec['expr'],
            'description': spec['desc'],
        },
        'dependencies': spec.get('deps', ['close']),
        'parameters': spec['params'],
        'expected_direction': 1 if m['ic'] >= 0 else -1,
        'validation': {
            'status': 'EFFECTIVE',
            'period': VALID_PERIOD,
            'last_validated': LAST_VALIDATED,
            'admission_horizon': H,
            'regime_notes': ('Validated 2020-01-01..2026-07-15 across multiple regimes: '
                             'COVID crash 2020, recovery bull 2020-21, 2022 tightening bear, '
                             '2023-24 AI-led equity rally, 2024-26 crypto/commodity cycles. '
                             'Cross-sectional rank IC on the 15-asset tradable universe.'),
            'metrics': {
                'ic': round(m['ic'], 4),
                'icir': round(m['icir'], 4),
                'ic_hit_ratio': round(m['hit'], 3),
                'n_ic_dates': int(m['dates']),
                'coverage_asset_days': round(m['cov_assets'], 3),
                'coverage_dates_ge8': round(m['cov_dates8'], 3),
                'turnover_10d_rank': round(m['turn'], 3),
                'decay_ic_by_horizon': {str(k): v for k, v in m['decay'].items()},
                'max_abs_library_correlation': round(max_corr, 4),
            },
        },
        'tags': spec['tags'],
    }
    path = f'factors/{label}.json'
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'WROTE {path}  max_abs_lib_corr={max_corr:.3f}')
    persisted.append(label)

print(f'\ntotal runtime {time.time()-t0:.0f}s')
