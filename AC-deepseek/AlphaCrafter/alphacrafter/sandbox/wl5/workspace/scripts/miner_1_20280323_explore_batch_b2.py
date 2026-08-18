# -*- coding: utf-8 -*-
"""miner_1 2028-03-23: second candidate batch (untested families).
Visible data through 2028-03-22. Gates: |IC| >= 0.007, |ICIR| >= 0.084 @10d,
rho < 0.5. Focus: volume dynamics, price z-score mean reversion, term-spread
rate beta, momentum acceleration, profit-factor asymmetry, dd speed.
"""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

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
print('Panel: %s -> %s | %d dates x %d assets'
      % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))


def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def build(name):
    if name == 'vol_ratio_20_60':
        # volume momentum: recent volume vs 60d baseline (liquidity/attention trend)
        return safe_div(V.rolling(20).mean(), V.rolling(60).mean())
    if name == 'vol_zscore_20_252_v':
        # volume z-score vs 1y history (abnormal activity)
        v20 = V.rolling(20).mean()
        mu = v20.rolling(252).mean()
        sd = v20.rolling(252).std()
        return safe_div(v20 - mu, sd)
    if name == 'price_zscore_60':
        # close vs 60d mean in units of 60d vol (deviation / mean reversion)
        mu = C.rolling(60).mean()
        sd = R.rolling(60).std()
        return safe_div(C - mu, sd)
    if name == 'term_spread_beta_60':
        # beta of asset returns to US10Y-CN10Y spread change (global rates divergence)
        sp = (C['US10Y'] - C['CN10Y']).pct_change()
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            cov = R[s].rolling(60).cov(sp)
            var = sp.rolling(60).var()
            out[s] = safe_div(cov, var)
        return out
    if name == 'mom_accel_20_60':
        # momentum acceleration: avg daily ret last 20d minus avg daily ret days 20-60
        r20 = C.pct_change(20)
        r60 = C.pct_change(60)
        a20 = r20 / 20.0
        a60 = (r60 - r20) / 40.0
        return safe_div(a20 - a60, R.rolling(60).std())
    if name == 'profit_factor_20':
        # sum of gains / |sum of losses| over 20d (trend quality of path)
        up = R.where(R > 0, 0.0).rolling(20).sum()
        dn = R.where(R < 0, 0.0).rolling(20).sum().abs()
        return safe_div(up, dn)
    if name == 'dd_speed_60':
        # drawdown depth per day underwater (fast vs slow drawdown)
        dd = C / C.cummax() - 1.0
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            d = dd[s]
            cnt = 0
            vals = []
            for v in d.values:
                cnt = cnt + 1 if v < 0 else 0
                vals.append(cnt)
            tuw = pd.Series(vals, index=d.index).replace(0, np.nan)
            out[s] = safe_div(d, tuw)
        return out
    if name == 'hl_pos_10':
        # short-horizon stochastic position in 10d range
        hi = H.rolling(10).max()
        lo = Lw.rolling(10).min()
        rng = (hi - lo).replace(0, np.nan)
        return safe_div(C - lo, rng)
    return None


CANDIDATES = ['vol_ratio_20_60', 'vol_zscore_20_252_v', 'price_zscore_60',
              'term_spread_beta_60', 'mom_accel_20_60', 'profit_factor_20',
              'dd_speed_60', 'hl_pos_10']

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C),
       'n_assets': C.shape[1], 'library_factors': lib_factors, 'results': {}}

for name in CANDIDATES:
    try:
        fp = build(name)
        if fp is None:
            print('\n[%s] build failed' % name)
            continue
        summ = L.full_validate(fp, R, horizon=10, label=name)
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        continue
    gate = (abs(summ['ic']) >= 0.007) and (abs(summ['icir']) >= 0.084)
    rho_ok = summ['max_abs_library_correlation'] < 0.5
    summ['pass_gate'] = bool(gate)
    summ['rho_ok'] = bool(rho_ok)
    s_ic = None
    try:
        fr = R.shift(-10)
        from miner3_lib import rank_ic
        s_ic = rank_ic(fp, fr)
    except Exception:
        pass
    recent = None
    if s_ic is not None and len(s_ic) >= 20:
        sub = s_ic[s_ic.index >= '2026-03-23']
        if len(sub) >= 20:
            recent = {'ic': round(float(sub.mean()), 4),
                      'icir': round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                      'n': int(len(sub))}
    summ['recency_2y'] = recent
    out['results'][name] = {k: summ[k] for k in
                            ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio',
                             'n_ic_dates', 'regime', 'coverage_asset_days',
                             'coverage_dates_ge8', 'turnover_10d_rank',
                             'decay_ic_by_horizon', 'max_abs_library_correlation',
                             'pass_gate', 'rho_ok', 'recency_2y']}
    print('\n=== %s ===' % name)
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'],
             summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  recency_2y:', recent)
    print('  max_abs_library_corr=%.3f' % summ['max_abs_library_correlation'])
    print('  GATE PASS: %s (ic=%s icir=%s rho_ok=%s)'
          % (gate and rho_ok, gate, abs(summ['icir']) >= 0.084, rho_ok))

with open('scripts/miner_1_20280323_explore_results_b2.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('\nSaved scripts/miner_1_20280323_explore_results_b2.json')
