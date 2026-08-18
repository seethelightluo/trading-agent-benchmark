# -*- coding: utf-8 -*-
"""miner_1 2028-03-23 batch b4: fix b3 bugs + robustness variants.
- up_down_vol_60: use semideviation squares (always defined) instead of NaN-heavy where().
- ret_autocorr_10: fix rolling.apply corr via np.corrcoef.
- add: semi_up_ratio_60, zscore_breakout_20, vol_skew_20.
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


def semi_vol(x, side, window):
    """side='up': sqrt(mean(max(r,0)^2)); side='dn': sqrt(mean(min(r,0)^2))."""
    if side == 'up':
        m = x.clip(lower=0)
    else:
        m = x.clip(upper=0)
    return np.sqrt((m ** 2).rolling(window).mean())


def build(name):
    if name == 'up_down_vol_60':
        upv = semi_vol(R, 'up', 60)
        dnv = semi_vol(R, 'dn', 60)
        return safe_div(dnv, upv)
    if name == 'semi_up_ratio_60':
        upv = semi_vol(R, 'up', 60)
        dnv = semi_vol(R, 'dn', 60)
        return safe_div(upv, dnv)
    if name == 'ret_autocorr_10':
        out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
        for s in R.columns:
            x = R[s].values
            ac = np.full(len(x), np.nan)
            for i in range(10, len(x)):
                w = x[i - 10:i]
                if np.isfinite(w).all() and w.std() > 0 and w[:-1].std() > 0 and w[1:].std() > 0:
                    ac[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
            out[s] = ac
        return out
    if name == 'zscore_breakout_20':
        mu = C.rolling(20).mean()
        sd = C.rolling(20).std()
        z = safe_div(C - mu, sd)
        return z.rolling(20).apply(lambda w: (w > 1.5).sum() - (w < -1.5).sum(), raw=True)
    if name == 'vol_skew_20':
        # heavy-tail asymmetry: mean(|r| above 75th pct) / mean(|r| below 25th pct)
        a = R.abs()
        hi = a.rolling(20).quantile(0.75)
        lo = a.rolling(20).quantile(0.25)
        hi_mean = a.where(a >= hi).rolling(20).mean()
        lo_mean = a.where(a <= lo).rolling(20).mean()
        return safe_div(hi_mean, lo_mean)
    return None


CANDIDATES = ['up_down_vol_60', 'semi_up_ratio_60', 'ret_autocorr_10',
              'zscore_breakout_20', 'vol_skew_20']

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

with open('scripts/miner_1_20280323_explore_results_b4.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20280323_explore_results_b4.json')
