# -*- coding: utf-8 -*-
"""miner_1 2028-03-23 batch b6: final robustness re-check.
b2 showed vol_ratio_20_60 with ic=+0.1088/icir=+0.4111 but n=4 IC dates only —
likely the same sparse-window artifact as b4's up_down_vol_60 (default min_periods
on a 7-day union calendar). Re-test with proper min_periods and add a few
remaining variants (vol_ratio_5_20, quantile skew 60, vol_of_ret 20 vs 252).
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
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))


def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def build(name):
    if name == 'vol_ratio_20_60':
        v20 = R.rolling(20, min_periods=10).std()
        v60 = R.rolling(60, min_periods=30).std()
        return safe_div(v20, v60)
    if name == 'vol_ratio_5_20':
        v5 = R.rolling(5, min_periods=3).std()
        v20 = R.rolling(20, min_periods=10).std()
        return safe_div(v5, v20)
    if name == 'quantile_skew_60':
        # (P90-P50)-(P50-P10) over 60d, normalized by (P90-P10): asymmetry of return dist
        p90 = R.rolling(60, min_periods=30).quantile(0.90)
        p50 = R.rolling(60, min_periods=30).quantile(0.50)
        p10 = R.rolling(60, min_periods=30).quantile(0.10)
        num = (p90 - p50) - (p50 - p10)
        den = (p90 - p10).replace(0, np.nan)
        return safe_div(num, den)
    if name == 'skew_60_mp':
        return R.rolling(60, min_periods=30).skew()
    return None


CANDIDATES = ['vol_ratio_20_60', 'vol_ratio_5_20', 'quantile_skew_60', 'skew_60_mp']

out = {'visible_through': str(C.index.max().date()), 'n_dates': len(C), 'n_assets': C.shape[1],
       'library_factors': L.LIB_FACTORS, 'results': {}}
for name in CANDIDATES:
    try:
        fp = build(name)
        if fp is None:
            print('\n[%s] build failed' % name)
            continue
        summ = L.full_validate(fp, R, horizon=10, label=name)
        ic, icir = summ['ic'], summ['icir']
        gate = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
        maxrho = summ['max_abs_library_correlation']
        rho_ok = maxrho < 0.5
        summ['pass_gate'] = bool(gate)
        summ['rho_ok'] = bool(rho_ok)
        out['results'][name] = {k: summ[k] for k in
                                ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                 'regime', 'coverage_asset_days', 'coverage_dates_ge8',
                                 'turnover_10d_rank', 'decay_ic_by_horizon',
                                 'max_abs_library_correlation', 'pass_gate', 'rho_ok']}
        print('%-22s ic=%+.4f icir=%+.4f n=%5d hit=%.3f cov=%.3f rho=%.3f gate=%s rho_ok=%s'
              % (name, ic, icir, summ['n_ic_dates'], summ['ic_hit_ratio'],
                 summ['coverage_dates_ge8'], maxrho, gate, rho_ok))
        print('    decay:', summ['decay_ic_by_horizon'])
        print('    regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        out['results'][name] = {'error': str(e)}

with open('scripts/miner_1_20280323_explore_results_b6.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20280323_explore_results_b6.json')
