"""miner_1 2026-07-30: persist candidates that passed the admission gate.

Passers (validated 2026-07-30, h=10, |IC|>=0.007 & |ICIR|>=0.084 & rho<0.5):
  upper_wick_10   IC=+0.0325 ICIR=+0.0998 rho=0.033 (NEW)
  max_ret_20d     IC=+0.0364 ICIR=+0.1071 rho=0.151 (NEW)
  bw_zscore_20_60 IC=+0.0429 ICIR=+0.1364 rho=0.071 (NEW)
  skew_term_20_60 IC=+0.0271 ICIR=+0.0886 rho=0.083 (NEW)
  mom_120d_skip5  IC=+0.0467 ICIR=+0.1228 rho=0.333 (RESTORE w/ artifact)
  vol_of_vol20x60 IC=+0.0409 ICIR=+0.1161 rho=0.045 (RESTORE w/ artifact)
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           persist_factor, canonical_grid, WATCHLIST, VAL_START, VAL_END)

t0 = time.time()
prices = load_prices(days=2200)
dxy = load_index('DXY', prices=prices)
vix = load_index('VIX', prices=prices)
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; grid {len(grid)} dates {grid.min().date()}..{grid.max().date()}', flush=True)

# fast validation identical to the screening batch
def rank_ic_series_fast(factor_panel, fwd_ret, min_valid=8):
    df = pd.concat({'x': factor_panel, 'y': fwd_ret}, axis=1).sort_index()
    x = df['x'].rank(axis=1); y = df['y'].rank(axis=1)
    valid = df['x'].notna() & df['y'].notna() & np.isfinite(df['x']) & np.isfinite(df['y'])
    n = valid.sum(axis=1)
    x = x.where(valid); y = y.where(valid)
    mx = x.mean(axis=1); my = y.mean(axis=1)
    cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
    vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
    vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(vx * vy)
    return ic.where((n >= min_valid) & (vx > 0) & (vy > 0)).dropna()

def forward_returns_fast(prices, horizon):
    cols = {s: df['close'].shift(-horizon) / df['close'] - 1.0 for s, df in prices.items()}
    return pd.DataFrame(cols).sort_index()

def validate_fast(factor_panel, prices, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    fwd = {h: forward_returns_fast(prices, h) for h in horizons}
    ic_s = {h: rank_ic_series_fast(factor_panel, fwd[h], min_valid) for h in horizons}
    ic10 = ic_s[10][(ic_s[10].index >= VAL_START) & (ic_s[10].index <= VAL_END)]
    ic_mean = float(ic10.mean()); ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = factor_panel[(factor_panel.index >= VAL_START) & (factor_panel.index <= VAL_END)]
    total = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total if total else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {str(h): (float(ic_s[h].mean()) if len(ic_s[h]) else float('nan')) for h in horizons}
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
            'turnover_10d_rank': turn, 'decay_ic_by_horizon': decay}

def f_upper_wick_10(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    uw = (df['high'] - np.maximum(df['open'], df['close'])) / rng
    return uw.rolling(10).mean()
def f_max_ret_20(df, s):
    return df['close'].pct_change().rolling(20).max()
def f_bw_zscore(df, s):
    c = df['close']
    ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    bw = 2.0 * sd / ma
    mu = bw.rolling(60).mean(); s = bw.rolling(60).std()
    return (bw - mu) / s.replace(0, np.nan)
def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
def f_volvol(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()

FACTORS = {
    'upper_wick_10': dict(fn=f_upper_wick_10, name='Upper Wick Ratio 10d',
        expr='mean10((high - max(open,close)) / (high - low))',
        desc='Mean fraction of upper wick (supply pressure) in the daily range over 10 days. '
             'High values (persistent upper wicks) predict outperformance over 10-20d in this '
             'cross-asset universe; nearly orthogonal to the existing library (rho=0.03).',
        deps=['open', 'high', 'low', 'close'], params={'window': 10}, dirn=1,
        tags=['microstructure', 'supply-pressure', 'price-shape'],
        regime='2020-01..2026-07 warm-up; cross-asset regime mix (COVID, 2022 tightening, 2023-25 risk-on, crypto cycles)'),
    'max_ret_20d': dict(fn=f_max_ret_20, name='Max Daily Return 20d',
        expr='max20(pct_change(close))',
        desc='Largest single-day return over the past 20 days. Positive IC: assets with a recent '
             'strong up-day continue to outperform (trend/attention proxy) in the cross-asset set.',
        deps=['close'], params={'window': 20}, dirn=1,
        tags=['momentum', 'lottery', 'tail'],
        regime='2020-01..2026-07 warm-up; cross-asset regime mix'),
    'bw_zscore_20_60': dict(fn=f_bw_zscore, name='Bollinger Bandwidth Z-Score 20x60',
        expr='zscore60(2*STD20(close)/SMA20(close))',
        desc='Bollinger bandwidth (2*std20/ma20) standardized over 60 days. Positive z = volatility '
             'bandwidth expanding vs its own norm; predicts outperformance over 10d (vol-expansion momentum).',
        deps=['close'], params={'window': 20, 'norm_window': 60}, dirn=1,
        tags=['volatility', 'regime', 'breakout'],
        regime='2020-01..2026-07 warm-up; cross-asset regime mix'),
    'skew_term_20_60': dict(fn=f_skew_term, name='Skewness Term Structure 20-60',
        expr='skew20(pct_change(close)) - skew60(pct_change(close))',
        desc='Short-horizon return skewness minus long-horizon skewness. Positive term (recent skew '
             'rising relative to longer history) predicts mild outperformance over 10d; low rho to library.',
        deps=['close'], params={'short': 20, 'long': 60}, dirn=1,
        tags=['skewness', 'tail-risk', 'momentum'],
        regime='2020-01..2026-07 warm-up; cross-asset regime mix'),
    'mom_120d_skip5': dict(fn=f_mom120, name='Momentum 120d skip5',
        expr='close.shift(5)/close.shift(125) - 1',
        desc='120-day price momentum skipping the most recent 5 days (library factor restored with '
             'signal artifact after quarantine for missing artifact). Strong cross-asset trend signal.',
        deps=['close'], params={'lookback': 120, 'skip': 5}, dirn=1,
        tags=['momentum', 'trend', 'restore'],
        regime='2020-01..2026-07 warm-up; cross-asset regime mix'),
    'vol_of_vol20x60': dict(fn=f_volvol, name='Volatility of Volatility 20x60',
        expr='STD60(STD20(pct_change(close)))',
        desc='Volatility of 20-day realized volatility over 60 days (library factor restored with '
             'signal artifact). High vol-of-vol predicts outperformance over 10d.',
        deps=['close'], params={'vol_window': 20, 'outer_window': 60}, dirn=1,
        tags=['volatility', 'regime', 'restore'],
        regime='2020-01..2026-07 warm-up; cross-asset regime mix'),
}

persisted = []
for fid, spec in FACTORS.items():
    print('#' * 72, flush=True)
    print(f'[persist] {fid}', flush=True)
    panel = factor_to_panel(spec['fn'], prices)
    m = validate_fast(panel, prices)
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f'  IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'cov={m["coverage_asset_days"]:.2f} turn={m["turnover_10d_rank"]:.2f} ndates={m["n_ic_dates"]} '
          f'-> {"PASS" if ok else "FAIL"}', flush=True)
    if not ok:
        print(f'  SKIP {fid}: no longer passes gate at persist time', flush=True)
        continue
    m['max_abs_library_correlation'] = spec['rho'] if 'rho' in spec else None
    path, arr = persist_factor(
        factor_id=fid, factor_name=spec['name'], expression=spec['expr'],
        description=spec['desc'], dependencies=spec['deps'], parameters=spec['params'],
        expected_direction=spec['dirn'], panel=panel, metrics=m, tags=spec['tags'],
        grid=grid, prices=prices, version='1.0.0', status='EFFECTIVE',
        regime_notes=spec['regime'])
    print(f'  wrote {path} artifact={arr.shape}', flush=True)
    persisted.append(fid)

print('=' * 72)
print('PERSISTED:', persisted)

# ---------- read-back verification ----------
print('\n--- READ-BACK VERIFICATION ---')
for fid in persisted:
    d = json.load(open(f'factors/{fid}.json'))
    art = np.load(f'factors/{fid}_signal.npy')
    m = d['validation']['metrics']
    assert d['factor_id'] == fid, 'id mismatch'
    assert d['validation']['status'] == 'EFFECTIVE', 'status mismatch'
    assert abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, 'gate mismatch'
    assert d.get('signal_artifact') is not None and art.shape == (len(grid), 15), 'artifact missing'
    print(f'OK {fid}: status={d["validation"]["status"]} IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} '
          f'rho={m.get("max_abs_library_correlation")} art={art.shape} last_validated={d["validation"]["last_validated"]}')
print(f'[total] {time.time()-t0:.1f}s')
