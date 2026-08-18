# -*- coding: utf-8 -*-
"""miner_3 2028-03-23: explore novel factor batch v2 - TRADING-CALENDAR windows.
Data through 2028-03-22. Gates: |IC| >= 0.007 and |ICIR| >= 0.084 at 10d horizon.
Robustness sanity: n_ic_dates >= 120 and coverage_dates_ge8 >= 0.6.
Key fix vs v1: union panel contains weekends -> rolling windows on union calendar
are NaN-fragile. All rolling stats are computed on each asset's OWN trading calendar
(dropna -> rolling -> reindex). IC forward returns stay on the union calendar (endpoint based).
Usage: python scripts/miner_3_20280323_explore_batch_v2.py A|B
"""
import sys, json, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

GROUP = sys.argv[1] if len(sys.argv) > 1 else 'A'

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


DXY = load_macro('DXY')
USDJPY = load_macro('USDJPY')
VIX = load_macro('VIX')


def tcal_roll(x, win, fn, minp=None):
    """Native rolling method on each asset's own trading calendar, reindexed to union index."""
    if minp is None:
        minp = win
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        v = x[s].dropna()
        r = getattr(v.rolling(win, min_periods=minp), fn)()
        out[s] = r.reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_roll_masked(x, win, fn, cond, minp):
    """Rolling method on trading calendar with a boolean mask applied after dropna."""
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        v = x[s].dropna()
        m = v.where(cond.reindex(v.index))
        r = getattr(m.rolling(win, min_periods=minp), fn)()
        out[s] = r.reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_shift(df, n):
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for s in df.columns:
        v = df[s].dropna()
        out[s] = v.shift(n).reindex(df.index)
    return out


def tcal_ret(df, n):
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for s in df.columns:
        v = df[s].dropna()
        out[s] = (v / v.shift(n) - 1.0).reindex(df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_change(x, n):
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        v = x[s].dropna()
        out[s] = (v - v.shift(n)).reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_beta(x, f, win, minp=None, cond=None):
    if minp is None:
        minp = win
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        a = x[s].dropna()
        b = f.reindex(a.index)
        b = b.dropna()
        a = a.reindex(b.index)
        if cond is not None:
            c = cond.reindex(a.index)
            a = a.where(c)
            b = b.where(c)
        cov = a.rolling(win, min_periods=minp).cov(b)
        var = b.rolling(win, min_periods=minp).var()
        out[s] = (cov / var).reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_corr(x, f, win, minp=None):
    if minp is None:
        minp = win
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        a = x[s].dropna()
        b = f.reindex(a.index).dropna()
        a = a.reindex(b.index)
        out[s] = a.rolling(win, min_periods=minp).corr(b).reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_acf(x, win=20):
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        v = x[s].dropna()
        xl = v.shift(1)
        ss = v.rolling(win).sum()
        sl = xl.rolling(win).sum()
        s2 = (v ** 2).rolling(win).sum()
        cr = (v * xl).rolling(win).sum()
        num = win * cr - ss * sl
        den = win * s2 - ss ** 2
        out[s] = (num / den).reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tcal_ewma(x, win=20, hl=5.0):
    w = np.exp(-np.log(2) * np.arange(1, win + 1) / hl)
    w = w / w.sum()
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        v = x[s].dropna()
        r = v.rolling(win).apply(lambda a: np.dot(a, w), raw=True)
        out[s] = r.reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)


M20 = tcal_ret(C, 20)
M60 = tcal_ret(C, 60)
VOL20 = tcal_roll(R, 20, 'std')
VOL60 = tcal_roll(R, 60, 'std')
EWIDX = R.mean(axis=1)
RB60 = tcal_beta(R, R['US10Y'], 60)
SPREAD = C['US10Y'] - C['CN10Y']
SH20 = tcal_shift(C, 20)


def build(name):
    if name == 'eff_ratio_20_signed':
        path = tcal_roll(R.abs(), 20, 'sum')
        net = (C - SH20).abs()
        eff = (net / path).replace([np.inf, -np.inf], np.nan)
        return eff * np.sign(M20)
    if name == 'max_gain_loss_20':
        g = tcal_roll(R, 20, 'max')
        l = tcal_roll(R, 20, 'min')
        return (g / l.abs()).replace([np.inf, -np.inf], np.nan)
    if name == 'updown_ratio_60':
        pos = tcal_roll_masked(R, 60, 'mean', R > 0, 25)
        neg = tcal_roll_masked(R, 60, 'mean', R < 0, 25)
        return (pos / neg.abs()).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_ts_5_60':
        return (tcal_roll(R, 5, 'std') / VOL60).replace([np.inf, -np.inf], np.nan)
    if name == 'vol_ts_10_60':
        return (tcal_roll(R, 10, 'std') / VOL60).replace([np.inf, -np.inf], np.nan)
    if name == 'uni_corr_60':
        return tcal_corr(R, EWIDX, 60)
    if name == 'downside_beta_60':
        return tcal_beta(R, EWIDX, 60, minp=30, cond=EWIDX < 0)
    if name == 'chn_beta_60':
        return tcal_beta(R, R['000300.SH'], 60)
    if name == 'rate_beta_60':
        return RB60
    if name == 'jpy_beta_60':
        return tcal_beta(R, USDJPY.pct_change(), 60)
    if name == 'dxy_up_beta_60':
        fd = DXY.pct_change()
        return tcal_beta(R, fd, 60, minp=30, cond=fd > 0)
    if name == 'rel_mom_vol_20':
        m = M20.sub(M20.median(axis=1), axis=0)
        return (m / VOL20).replace([np.inf, -np.inf], np.nan)
    if name == 'hi_prox_20':
        hi = tcal_roll(C, 20, 'max')
        return (C / hi - 1.0).replace([np.inf, -np.inf], np.nan)
    if name == 'skew_60_signed':
        return tcal_roll(R, 60, 'skew')
    if name == 'dd_speed_60x10':
        hi = tcal_roll(C, 60, 'max')
        dd = (C / hi - 1.0).replace([np.inf, -np.inf], np.nan)
        return tcal_change(dd, 10)
    if name == 'ewma_mom_20':
        return tcal_ewma(R, 20, 5.0)
    if name == 'ser_corr_20':
        return tcal_acf(R, 20)
    if name == 'volwt_mom_20':
        vt = tcal_roll(V, 20, 'mean') / tcal_roll(V, 60, 'mean').replace(0, np.nan)
        return (M20 * vt).replace([np.inf, -np.inf], np.nan)
    if name == 'cond_mom_20_60':
        ma60 = tcal_roll(C, 60, 'mean')
        return M20.where(C > ma60, M20 * 0.5)
    if name == 'rate_carry_20':
        sp_chg = SPREAD.dropna()
        chg = (sp_chg - sp_chg.shift(20)).reindex(C.index)
        return (RB60 * chg).replace([np.inf, -np.inf], np.nan)
    if name == 'crypto_beta_60':
        return tcal_beta(R, R['BTC'], 60)
    if name == 'max_dd_120':
        hi = tcal_roll(C, 120, 'max')
        dd = (C / hi - 1.0).replace([np.inf, -np.inf], np.nan)
        return tcal_roll(dd, 120, 'min')
    if name == 'disp_tilt_20':
        med = M20.median(axis=1)
        return (M20 - med).abs()
    if name == 'rel_mom_20':
        return M20.sub(M20.median(axis=1), axis=0)
    if name == 'mom_accel_20_60':
        return (M20 - M60).replace([np.inf, -np.inf], np.nan)
    return None


GROUP_A = ['eff_ratio_20_signed', 'max_gain_loss_20', 'updown_ratio_60',
           'vol_ts_5_60', 'vol_ts_10_60', 'uni_corr_60', 'downside_beta_60',
           'chn_beta_60', 'rate_beta_60', 'jpy_beta_60', 'dxy_up_beta_60',
           'rel_mom_vol_20']
GROUP_B = ['hi_prox_20', 'skew_60_signed', 'dd_speed_60x10', 'ewma_mom_20',
           'ser_corr_20', 'volwt_mom_20', 'cond_mom_20_60', 'rate_carry_20',
           'crypto_beta_60', 'max_dd_120', 'disp_tilt_20', 'rel_mom_20',
           'mom_accel_20_60']

CANDIDATES = GROUP_A if GROUP == 'A' else GROUP_B
print('Group %s: %d candidates' % (GROUP, len(CANDIDATES)))

results = {}
for name in CANDIDATES:
    t0 = time.time()
    fp = build(name)
    if fp is None:
        print('\n[%s] build failed' % name)
        continue
    try:
        summ = L.full_validate(fp, R, horizon=10, label=name)
    except Exception as e:
        print('\n[%s] validation error: %s' % (name, e))
        continue
    n_ok = summ['n_ic_dates'] >= 120
    cov_ok = summ['coverage_dates_ge8'] >= 0.6
    gate_ic = abs(summ['ic']) >= 0.007
    gate_icir = abs(summ['icir']) >= 0.084
    robust = n_ok and cov_ok
    results[name] = summ
    print('\n=== %s (%.1fs) ===' % (name, time.time() - t0))
    print('  IC=%.4f ICIR=%.4f hit=%.3f n=%d cov_asset=%.3f cov_dates_ge8=%.3f turn=%.3f'
          % (summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
             summ['coverage_asset_days'], summ['coverage_dates_ge8'], summ['turnover_10d_rank']))
    print('  decay:', summ['decay_ic_by_horizon'])
    print('  regime:', {k: v['ic'] for k, v in summ.get('regime', {}).items()})
    print('  max_abs_library_corr=%.3f' % summ['max_abs_library_correlation'])
    print('  GATE: ic=%s icir=%s robust(n>=120,cov>=0.6)=%s  => PASS=%s'
          % (gate_ic, gate_icir, robust, gate_ic and gate_icir and robust))

out = {'visible_through': str(C.index.max().date()), 'n_dates': int(len(C)),
       'n_assets': int(C.shape[1]), 'group': GROUP, 'library_factors': lib_factors,
       'results': {k: {kk: vv for kk, vv in v.items() if kk != 'library_rho_by_factor'}
                   for k, v in results.items()}}
with open('scripts/miner_3_20280323_explore_results_%s.json' % GROUP, 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved results to scripts/miner_3_20280323_explore_results_%s.json' % GROUP)
