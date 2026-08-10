"""miner_2 round-8 screening: novel factor candidates on the 15-asset cross-asset universe.

Family A (conditional macro beta, extending the DXY/EURUSD/VIX cond-beta family):
  - usdjpy_beta_cond_60x20, usdcny_beta_cond_60x20, xau_beta_cond_60x20,
    wti_beta_cond_60x20, cn10y_beta_cond_60x20, btc_beta_cond_60x20, eth_beta_cond_60x20
Family B (structural, low correlation with beta/momentum library):
  - yield_spread_beta_60x20 : beta to (US10Y-CN10Y) spread change
  - crisis_beta_60          : beta to SPX on high-vol-regime days
  - amihud_illiq_20d        : |ret|/volume liquidity factor
  - gk_vol_ratio_10_60      : Garman-Klass vol term structure
  - downside_vol_ratio_60   : semideviation share of total vol
  - dd_depth_60             : drawdown depth (distance from 60d high)
"""
import sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, WATCHLIST, VAL_START, VAL_END)

t0 = time.time()
prices = load_prices(days=2000)
print(f'loaded {len(prices)} assets; max date {max(d.index.max() for d in prices.values())}')

vix = load_index('VIX', prices=prices)
dxy = load_index('DXY', prices=prices)
usdjpy = load_index('USDJPY', prices=prices)
usdcny = load_index('USDCNY', prices=prices)
print('index signals loaded:', {k: (None if v is None else len(v)) for k, v in
      [('VIX', vix), ('DXY', dxy), ('USDJPY', usdjpy), ('USDCNY', usdcny)]})

def _cond_beta_factory(cond_series, name):
    def fn(df, s):
        if cond_series is None or len(cond_series) < 90:
            return None
        r = df['close'].pct_change()
        rc = cond_series['close'].pct_change()
        z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
        beta = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()
        move = cond_series['close'] / cond_series['close'].shift(20) - 1.0
        return (beta * move).reindex(z.index)
    return fn

def yield_spread_beta_60x20(df, s):
    u10 = prices.get('US10Y'); c10 = prices.get('CN10Y')
    if u10 is None or c10 is None:
        return None
    spread = u10['close'] - c10['close']
    r = df['close'].pct_change()
    rs = spread.pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()
    move = spread / spread.shift(20) - 1.0
    return (beta * move).reindex(z.index)

def crisis_beta_60(df, s):
    """Beta to SPX computed only on days when SPX 20d vol > its 1y rolling median (crisis regime)."""
    spx = prices.get('SPX')
    if spx is None:
        return None
    r = df['close'].pct_change()
    rs = spx['close'].pct_change()
    spx_vol = rs.rolling(20).std()
    med = spx_vol.rolling(252).median()
    crisis = (spx_vol > med).astype(float)
    z = pd.concat([r.rename('r'), rs.rename('s'), crisis.rename('c')], axis=1).dropna()
    out = pd.Series(np.nan, index=z.index)
    # rolling beta over 60d using only crisis-day returns via expanding cov weighting
    cr = z[z['c'] > 0][['r', 's']]
    b = cr['r'].rolling(60).cov(cr['s']) / cr['s'].rolling(60).var()
    out.loc[b.index] = b
    return out

def amihud_illiq_20d(df, s):
    ret = df['close'].pct_change().abs()
    vol = df['volume'].replace(0, np.nan)
    illiq = (ret / vol).rolling(20).mean()
    return -illiq  # negative: higher liquidity -> higher factor (direction agnostic for IC)

def gk_vol_ratio_10_60(df, s):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    v10 = gk.rolling(10).mean().pow(0.5)
    v60 = gk.rolling(60).mean().pow(0.5)
    return v10 / v60

def downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(60).mean()
    dd = r[r < mu] - mu[r < mu]
    semi = (dd ** 2).rolling(60).mean().pow(0.5)
    total = r.rolling(60).std()
    return semi / total

def dd_depth_60(df, s):
    """Current drawdown depth relative to 60d rolling max close."""
    return df['close'] / df['close'].rolling(60).max() - 1.0

candidates = {
    'usdjpy_beta_cond_60x20': _cond_beta_factory(usdjpy, 'USDJPY'),
    'usdcny_beta_cond_60x20': _cond_beta_factory(usdcny, 'USDCNY'),
    'xau_beta_cond_60x20': _cond_beta_factory(prices.get('XAU'), 'XAU'),
    'wti_beta_cond_60x20': _cond_beta_factory(prices.get('WTI'), 'WTI'),
    'cn10y_beta_cond_60x20': _cond_beta_factory(prices.get('CN10Y'), 'CN10Y'),
    'btc_beta_cond_60x20': _cond_beta_factory(prices.get('BTC'), 'BTC'),
    'eth_beta_cond_60x20': _cond_beta_factory(prices.get('ETH'), 'ETH'),
    'yield_spread_beta_60x20': yield_spread_beta_60x20,
    'crisis_beta_60': crisis_beta_60,
    'amihud_illiq_20d': amihud_illiq_20d,
    'gk_vol_ratio_10_60': gk_vol_ratio_10_60,
    'downside_vol_ratio_60': downside_vol_ratio_60,
    'dd_depth_60': dd_depth_60,
}

IC_TH, ICIR_TH = 0.007, 0.084

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f'{fid}: insufficient data -> None', flush=True)
            continue
        ok_ic = abs(m['ic']) >= IC_TH and abs(m['icir']) >= ICIR_TH
        print(f'{fid}: ic={m["ic"]:+.4f} icir={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
              f'n={m["n_ic_dates"]} cov={m["coverage_asset_days"]:.3f} ge8={m["coverage_dates_ge8"]:.3f} '
              f'turn={m["turnover_10d_rank"]:.2f} decay10={m["decay_ic_by_horizon"]["10"]:+.4f} '
              f'decay20={m["decay_ic_by_horizon"]["20"]:+.4f} -> {"PASS-GATE" if ok_ic else "skip"} '
              f'[{time.time()-t1:.1f}s]', flush=True)
        results[fid] = (m, panel)
    except Exception as e:
        print(f'{fid}: ERROR {type(e).__name__}: {e}', flush=True)

print(f'\nTOTAL {time.time()-t0:.1f}s')
print('SUMMARY:')
for fid, (m, _) in sorted(results.items()):
    ok = abs(m['ic']) >= IC_TH and abs(m['icir']) >= ICIR_TH
    print(f'  {fid:28s} ic={m["ic"]:+.4f} icir={m["icir"]:+.4f} -> {"ADMIT" if ok else "skip"}')
