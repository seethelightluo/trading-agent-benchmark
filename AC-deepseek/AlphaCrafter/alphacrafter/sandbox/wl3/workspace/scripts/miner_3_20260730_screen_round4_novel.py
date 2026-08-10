"""miner_3 round-4 screening: novel factor ideas.

Ideas (all per-asset, cross-sectional over 15 tradable instruments):
 1. overnight_gap_20   : 20d mean of (open/prev_close - 1): attention/microstructure drift
 2. amihud_illiq_20    : 20d mean(|ret|/volume): illiquidity premium (volume family, absent in library)
 3. semi_vol_ratio_60  : downside vol / upside vol (semivariance asymmetry)
 4. parkinson_ratio_20 : Parkinson vol(20) / close-close vol(20): intraday vs overnight info share
 5. trend_r2_60        : R^2 of close vs linear time over 60d: trend consistency (not magnitude)
 6. basket_corr_60     : 60d rolling corr of asset return with equal-weight basket return
 7. min_ret_20d        : worst single daily return over 20d (complement of persisted max_ret_20d)
 8. updown_vol_asym_20 : (vol of up days - vol of down days)/(total vol)
 9. cn10y_beta_60      : 60d beta of asset to CN10Y yield change (bond beta, CN side)
10. us10y_beta_60      : 60d beta of asset to US10Y yield change (bond beta, US side)
11. gap_fill_ratio_20  : 20d mean of (close-open)/(high-low): close location within day (loc variant)
12. ret_skew_raw_60    : raw (not standardized) skewness of daily returns over 60d (vs skew_term_20_60)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from factor_common import (
    WATCHLIST, load_prices, load_index, canonical_grid, factor_to_panel,
    validate_factor, max_library_correlation, build_library_panels,
    load_artifact_matrix,
)

prices = load_prices(days=3000)
grid = canonical_grid(prices)
print(f'grid n={len(grid)} {grid.min().date()}..{grid.max().date()}')

# observation-only macro for bond beta candidates
cn10y = load_index('CN10Y', prices=prices)
us10y = load_index('US10Y', prices=prices)
print('cn10y index rows:', 0 if cn10y is None else len(cn10y),
      'us10y index rows:', 0 if us10y is None else len(us10y))

# equal-weight basket return for basket_corr_60
ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)
print(f'basket span {basket.index.min().date()}..{basket.index.max().date()}')

def overnight_gap_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    return gap.rolling(20).mean()

def amihud_illiq_20(df, s):
    vol = df['volume'].replace(0, np.nan).astype(float)
    illiq = (df['close'].pct_change().abs() / vol)
    return illiq.rolling(20).mean()

def semi_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    down = r.where(r < 0)
    up = r.where(r > 0)
    dv = down.rolling(60).std()
    uv = up.rolling(60).std()
    return (dv / uv.replace(0, np.nan))

def parkinson_ratio_20(df, s):
    c = df['close'].replace(0, np.nan)
    park = np.sqrt((np.log(df['high'] / df['low']) ** 2).rolling(20).mean() / (4 * np.log(2)))
    cc = c.pct_change().rolling(20).std()
    return park / cc.replace(0, np.nan)

def trend_r2_60(df, s):
    c = df['close']
    x = np.arange(60, dtype=float)
    out = pd.Series(np.nan, index=c.index)
    vals = c.values
    n = len(vals)
    if n < 60:
        return out
    sx, sx2 = x.sum(), (x ** 2).sum()
    denom = n * sx2 - sx * sx
    for i in range(60 - 1, n):
        w = vals[i - 59:i + 1]
        if not np.all(np.isfinite(w)):
            continue
        sy, sy2, sxy = w.sum(), (w ** 2).sum(), float((x * w).sum())
        if (n * sy2 - sy * sy) <= 0 or denom <= 0:
            continue
        r = (n * sxy - sx * sy) / np.sqrt(denom * (n * sy2 - sy * sy))
        out.iloc[i] = r * r
    return out

def basket_corr_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    return z['r'].rolling(60).corr(z['b'])

def min_ret_20d(df, s):
    return df['close'].pct_change().rolling(20).min()

def updown_vol_asym_20(df, s):
    r = df['close'].pct_change()
    down = r.where(r < 0)
    up = r.where(r > 0)
    dv = down.rolling(20).std()
    uv = up.rolling(20).std()
    tot = r.rolling(20).std().replace(0, np.nan)
    return (uv - dv) / tot

def beta_to_yield(df, s, ydf):
    if ydf is None:
        return None
    r = df['close'].pct_change()
    dy = ydf['close'].pct_change()
    z = pd.concat([r.rename('r'), dy.rename('y')], axis=1).dropna()
    cov = z['r'].rolling(60).cov(z['y'])
    var = z['y'].rolling(60).var().replace(0, np.nan)
    return cov / var

def cn10y_beta_60(df, s):
    return beta_to_yield(df, s, cn10y)

def us10y_beta_60(df, s):
    return beta_to_yield(df, s, us10y)

def gap_fill_ratio_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['open']) / rng).rolling(20).mean()

def ret_skew_raw_60(df, s):
    return df['close'].pct_change().rolling(60).skew()

candidates = {
    'overnight_gap_20': overnight_gap_20,
    'amihud_illiq_20': amihud_illiq_20,
    'semi_vol_ratio_60': semi_vol_ratio_60,
    'parkinson_ratio_20': parkinson_ratio_20,
    'trend_r2_60': trend_r2_60,
    'basket_corr_60': basket_corr_60,
    'min_ret_20d': min_ret_20d,
    'updown_vol_asym_20': updown_vol_asym_20,
    'cn10y_beta_60': cn10y_beta_60,
    'us10y_beta_60': us10y_beta_60,
    'gap_fill_ratio_20': gap_fill_ratio_20,
    'ret_skew_raw_60': ret_skew_raw_60,
}

# extended library: legacy 4 + all persisted artifact panels
library_panels = build_library_panels(prices)
for jp in sorted(Path('factors').glob('*.json')):
    if jp.name == 'factor_ensemble.json':
        continue
    art = load_artifact_matrix(str(jp))
    if art is None or art.shape[0] != len(grid):
        continue
    fid = json.loads(jp.read_text(encoding='utf-8')).get('factor_id')
    library_panels[fid] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
print('extended library:', sorted(library_panels.keys()))

for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None'); continue
    rho, best = max_library_correlation(panel, library_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"\n== {fid}: panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()}")
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str))
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str))
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}  max_corr={rho:.3f} vs {best}")
