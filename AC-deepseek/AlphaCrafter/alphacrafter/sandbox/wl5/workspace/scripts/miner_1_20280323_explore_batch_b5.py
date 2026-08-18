# -*- coding: utf-8 -*-
"""miner_1 2028-03-23 batch b5: fix min_periods for sparse daily data.
rolling(window).mean() default requires window non-NaN obs -> fails for 5-day/week
assets on a 7-day union calendar (only recent continuous stretch valid). Use
min_periods=~window/2 so values are defined across the full history, then
re-validate up_down_vol_60, semi_up_ratio_60, vol_skew_20, and a few variants.
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


def semi_vol(x, side, window, mp):
    if side == 'up':
        m = x.clip(lower=0)
    else:
        m = x.clip(upper=0)
    return np.sqrt((m ** 2).rolling(window, min_periods=mp).mean())


def build(name):
    if name == 'up_down_vol_60':
        upv = semi_vol(R, 'up', 60, 30)
        dnv = semi_vol(R, 'dn', 60, 30)
        return safe_div(dnv, upv)
    if name == 'semi_up_ratio_60':
        upv = semi_vol(R, 'up', 60, 30)
        dnv = semi_vol(R, 'dn', 60, 30)
        return safe_div(upv, dnv)
    if name == 'up_down_vol_20':
        upv = semi_vol(R, 'up', 20, 10)
        dnv = semi_vol(R, 'dn', 20, 10)
        return safe_div(dnv, upv)
    if name == 'vol_skew_20':
        a = R.abs()
        hi = a.rolling(20, min_periods=10).quantile(0.75)
        lo = a.rolling(20, min_periods=10).quantile(0.25)
        hi_mean = a.where(a >= hi).rolling(20, min_periods=10).mean()
        lo_mean = a.where(a <= lo).rolling(20, min_periods=10).mean()
        return safe_div(hi_mean, lo_mean)
    if name == 'neg_vol_share_60':
        # share of total semivariance coming from downside moves (0.5 = symmetric)
        upv2 = (R.clip(lower=0) ** 2).rolling(60, min_periods=30).mean()
        dnv2 = (R.clip(upper=0) ** 2).rolling(60, min_periods=30).mean()
        return safe_div(dnv2, upv2 + dnv2)
    if name == 'up_vol_60':
        return semi_vol(R, 'up', 60, 30)
    if name == 'dn_vol_60':
        return semi_vol(R, 'dn', 60, 30)
    return None


CANDIDATES = ['up_down_vol_60', 'semi_up_ratio_60', 'up_down_vol_20',
              'vol_skew_20', 'neg_vol_share_60']

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
        print('    rho_by:', {k: v for k, v in summ.get('library_rho_by_factor', {}).items()
                              if v is not None and abs(v) > 0.3})
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        out['results'][name] = {'error': str(e)}

with open('scripts/miner_1_20280323_explore_results_b5.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20280323_explore_results_b5.json')
