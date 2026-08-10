"""miner_1 2026-07-30: full re-validation of quarantined library factors + new candidates.

The library was quarantined solely because persisted JSONs lacked recoverable
signal artifacts. This script recomputes metrics for (a) the 4 known-effective
library factors, (b) hilo_pos_20d, and (c) several new candidate ideas, then
prints admission status. Passing factors are persisted in a follow-up step.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           evaluate_candidate, build_library_panels, VAL_START, VAL_END)

prices = load_prices(days=2200)
print('loaded assets:', len(prices))
print('date range:', min(df.index.min() for df in prices.values()).date(),
      '..', max(df.index.max() for df in prices.values()).date())

# --- index signals (observation-only) ---
usdjpy = load_index('USDJPY', prices=prices)
dxy = load_index('DXY', prices=prices)
vix = load_index('VIX', prices=prices)
print('signals: USDJPY', None if usdjpy is None else (usdjpy.index.min().date(), usdjpy.index.max().date(), len(usdjpy)),
      'VIX', None if vix is None else len(vix))

# ============ A. LIBRARY RESTORE CANDIDATES ============
def f_mom10(df, s): return df['close'].shift(5) / df['close'].shift(15) - 1.0
def f_mom120(df, s): return df['close'].shift(5) / df['close'].shift(125) - 1.0
def f_vixbeta(df, s):
    if vix is None: return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
def f_volvol(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
def f_hilo20(df, s):
    hi = df['high'].rolling(20).max(); lo = df['low'].rolling(20).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)

# ============ B. NEW CANDIDATES ============
def f_usdjpy_beta(df, s):
    """Beta to USDJPY x USDJPY 20d move (JPY carry / risk-appetite proxy)."""
    if usdjpy is None: return None
    r = df['close'].pct_change(); ru = usdjpy['close'].pct_change()
    z = pd.concat([r.rename('r'), ru.rename('u')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['u']) / z['u'].rolling(60).var()
    move = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)

def f_xau_beta(df, s):
    """Beta to XAU x XAU 20d move (safe-haven / real-asset beta)."""
    xau = prices.get('XAU')
    if xau is None: return None
    r = df['close'].pct_change(); rx = xau['close'].pct_change()
    z = pd.concat([r.rename('r'), rx.rename('x')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var()
    move = xau['close'] / xau['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)

def f_wti_beta(df, s):
    """Beta to WTI x WTI 20d move (energy/commodity beta)."""
    wti = prices.get('WTI')
    if wti is None: return None
    r = df['close'].pct_change(); rw = wti['close'].pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['w']) / z['w'].rolling(60).var()
    move = wti['close'] / wti['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)

def f_accel(df, s):
    """Trend acceleration: 20d momentum minus 60d momentum (skip 5)."""
    m20 = df['close'].shift(5) / df['close'].shift(25) - 1.0
    m60 = df['close'].shift(5) / df['close'].shift(65) - 1.0
    return m20 - m60

def f_vol_z20(df, s):
    """Volatility z-score: (vol20 - mean(vol20,60)) / std(vol20,60)."""
    v = df['close'].pct_change().rolling(20).std()
    mu = v.rolling(60).mean(); sd = v.rolling(60).std()
    return (v - mu) / sd.replace(0, np.nan)

def f_skew_60(df, s):
    """Rolling 60d return skewness."""
    return df['close'].pct_change().rolling(60).skew()

def f_dd_60(df, s):
    """Drawdown vs 60d high (negative: far below high)."""
    hi = df['close'].rolling(60).max()
    return df['close'] / hi - 1.0

def f_eff_ratio_60(df, s):
    """Kaufman efficiency ratio 60d."""
    n = 60
    num = (df['close'] - df['close'].shift(n)).abs()
    den = df['close'].pct_change().abs().rolling(n).sum()
    return num / den.replace(0, np.nan)

def f_mom20_risk_adj(df, s):
    """Risk-adjusted momentum 20d / vol20."""
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(20).std()
    return m / v.replace(0, np.nan)

def f_zscore_20(df, s):
    """Bollinger z-score 20d (mean reversion)."""
    m = df['close'].rolling(20).mean(); sd = df['close'].rolling(20).std()
    return (df['close'] - m) / sd.replace(0, np.nan)

CANDIDATES = [
    ('mom_10d_skip5', f_mom10, 'LIBRARY RESTORE'),
    ('mom_120d_skip5', f_mom120, 'LIBRARY RESTORE'),
    ('vix_beta_cond_60x20', f_vixbeta, 'LIBRARY RESTORE'),
    ('vol_of_vol20x60', f_volvol, 'LIBRARY RESTORE'),
    ('hilo_pos_20d', f_hilo20, 'RESTORE'),
    ('usdjpy_beta_cond_60x20', f_usdjpy_beta, 'NEW'),
    ('xau_beta_cond_60x20', f_xau_beta, 'NEW'),
    ('wti_beta_cond_60x20', f_wti_beta, 'NEW'),
    ('accel_20x60_skip5', f_accel, 'NEW'),
    ('vol_zscore_20x60', f_vol_z20, 'NEW'),
    ('skew_60d', f_skew_60, 'NEW'),
    ('dd_60d', f_dd_60, 'NEW'),
    ('eff_ratio_60d', f_eff_ratio_60, 'NEW'),
    ('mom20_risk_adj', f_mom20_risk_adj, 'NEW'),
    ('zscore_20', f_zscore_20, 'NEW'),
]

results = {}
for fid, fn, kind in CANDIDATES:
    print('#' * 72)
    print(f'[{kind}] {fid}')
    m, panel = evaluate_candidate(fid, fn, prices)
    if m is not None:
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        results[fid] = {'metrics': m, 'panel': panel, 'ok': ok, 'kind': kind}
        print(f'FINAL: {fid} {"PASS" if ok else "FAIL"} |IC|={abs(m["ic"]):.4f} |ICIR|={abs(m["icir"]):.4f}')

print('=' * 72)
print('SUMMARY:')
for fid, r in results.items():
    m = r['metrics']
    print(f"  {fid:24s} {r['kind']:14s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} cov={m['coverage_asset_days']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={m['max_abs_library_correlation']:.3f} "
          f"-> {'PASS' if r['ok'] else 'FAIL'}")

with open('scripts/miner_1_20260730_results.json', 'w') as fh:
    json.dump({fid: {k: (v if k != 'metrics' else {kk: vv for kk, vv in v.items() if kk != 'decay_ic_by_horizon'})
                     for k, v in r.items() if k != 'panel'}
               for fid, r in results.items()}, fh, indent=2, default=str)
print('saved results to scripts/miner_1_20260730_results.json')
