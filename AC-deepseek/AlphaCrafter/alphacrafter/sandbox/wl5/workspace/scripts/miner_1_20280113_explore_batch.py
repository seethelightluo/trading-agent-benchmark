# -*- coding: utf-8 -*-
"""miner_1 2028-01-13: explore novel factor batch.
Visible data through 2028-01-12. Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d
horizon on the 15-asset universe; post-gate max_abs_library_correlation < 0.5.
Focus: trend efficiency, risk-adjusted return, momentum acceleration,
vol asymmetry, stochastic position, short reversal, yield beta, SPX beta asymmetry.
"""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# refresh library factor list from live factor files (EFFECTIVE + artifact)
lib_factors = []
for p in sorted(os.listdir('factors')):
    if not p.endswith('.json') or p == 'factor_ensemble.json':
        continue
    try:
        d = json.load(open('factors/' + p))
        if d.get('validation', {}).get('status') == 'EFFECTIVE' and \
           d.get('validation', {}).get('signal_artifact'):
            lib_factors.append(d['factor_id'])
    except Exception:
        pass
L.LIB_FACTORS = lib_factors
print('Library factors for rho check (%d): %s' % (len(lib_factors), lib_factors))

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

VIX = load_macro('VIX')

def build(name):
    # ---- trend efficiency ----
    if name == 'eff_ratio_60':
        net = (C / C.shift(60) - 1.0).abs()
        path = R.abs().rolling(60).sum()
        return (net / path).replace([np.inf, -np.inf], np.nan)
    if name == 'eff_ratio_20':
        net = (C / C.shift(20) - 1.0).abs()
        path = R.abs().rolling(20).sum()
        return (net / path).replace([np.inf, -np.inf], np.nan)
    # ---- risk-adjusted return ----
    if name == 'sharpe_60':
        mu = R.rolling(60).mean()
        sd = R.rolling(60).std()
        return (mu / sd).replace([np.inf, -np.inf], np.nan)
    # ---- momentum acceleration ----
    if name == 'mom_accel_20_60':
        r20 = C.pct_change(20)
        r60 = C.pct_change(60)
        return (r20 - r60).replace([np.inf, -np.inf], np.nan)
    # ---- vol asymmetry: downside vol / upside vol over 20d ----
    if name == 'down_up_vol_20':
        up = R.where(R > 0)
        dn = R.where(R < 0)
        upv = up.rolling(20).std()
        dnv = dn.rolling(20).std()
        return (dnv / upv).replace([np.inf, -np.inf], np.nan)
    # ---- stochastic position within 20d high-low range ----
    if name == 'stoch_pos_20':
        hi = H.rolling(20).max()
        lo = Lw.rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        return ((C - lo) / rng).replace([np.inf, -np.inf], np.nan)
    # ---- short-term reversal (5d) ----
    if name == 'reversal_5':
        return -(C.pct_change(5))
    # ---- yield beta 60d (beta to US10Y yield change) ----
    if name == 'us10y_beta_60':
        f = C['US10Y'].pct_change()
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            x = R[s]
            cov = x.rolling(60).cov(f)
            var = f.rolling(60).var()
            out[s] = (cov / var).replace([np.inf, -np.inf], np.nan)
        return out
    # ---- SPX up/down beta asymmetry ----
    if name == 'up_down_beta_60':
        f = R['SPX']
        upm = (f > 0).astype(float)
        dnm = (f < 0).astype(float)
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            x = R[s]
            covu = x.rolling(60).cov(f) * upm.rolling(60).mean()
            covd = x.rolling(60).cov(f) * dnm.rolling(60).mean()
            varu = f.rolling(60).var() * upm.rolling(60).mean()
            vard = f.rolling(60).var() * dnm.rolling(60).mean()
            bu = (covu / varu).replace([np.inf, -np.inf], np.nan)
            bd = (covd / vard).replace([np.inf, -np.inf], np.nan)
            out[s] = (bu - bd).replace([np.inf, -np.inf], np.nan)
        return out
    # ---- VIX trend regime filter for momentum (conditional) ----
    if name == 'mom20_vix_filter':
        vix_trend = (VIX > VIX.rolling(60).mean()).astype(float)
        r20 = C.pct_change(20)
        return (r20 * (1 - vix_trend).values).replace([np.inf, -np.inf], np.nan)
    return None

CANDIDATES = ['eff_ratio_60', 'eff_ratio_20', 'sharpe_60', 'mom_accel_20_60',
              'down_up_vol_20', 'stoch_pos_20', 'reversal_5', 'us10y_beta_60',
              'up_down_beta_60', 'mom20_vix_filter']

for name in CANDIDATES:
    fp = build(name)
    if fp is None:
        print('\n[%s] build failed' % name)
        continue
    try:
        summ = L.full_validate(fp, R, horizon=10, label=name)
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        continue
    gate_ic = abs(summ['ic']) >= 0.007
    gate_icir = abs(summ['icir']) >= 0.084
    print('\n=== %s ===' % name)
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'], summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  max_abs_library_corr=%.3f  rho_by_fac=%s' % (summ['max_abs_library_correlation'],
          {k: v for k, v in summ.get('library_rho_by_factor', {}).items() if v is not None}))
    print('  GATE PASS: %s (ic=%s icir=%s)' % (gate_ic and gate_icir, gate_ic, gate_icir))
