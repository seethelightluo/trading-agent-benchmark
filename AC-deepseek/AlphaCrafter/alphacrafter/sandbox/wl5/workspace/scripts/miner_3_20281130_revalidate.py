# -*- coding: utf-8 -*-
"""miner_3 2028-11-30: re-validate all EFFECTIVE library factors with data through 2028-11-29.
Gates: |IC| >= 0.007, |ICIR| >= 0.084 at 10d horizon. Reports drift vs stored metrics.
Does NOT persist anything; only diagnostics for the cycle.
"""
import sys, json, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# build library factor list from disk (EFFECTIVE with signal artifact OR without)
lib_factors = []
for p in sorted(os.listdir('factors')):
    if not p.endswith('.json') or p == 'factor_ensemble.json':
        continue
    try:
        d = json.load(open('factors/' + p))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            lib_factors.append(d['factor_id'])
    except Exception:
        pass
L.LIB_FACTORS = lib_factors
print('Library factors (%d): %s' % (len(lib_factors), lib_factors))

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

def tcal_roll(x, win, fn, minp=None):
    if minp is None:
        minp = win
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        v = x[s].dropna()
        r = getattr(v.rolling(win, min_periods=minp), fn)()
        out[s] = r.reindex(x.index)
    return out.replace([np.inf, -np.inf], np.nan)

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY')
USDJPY = load_macro('USDJPY')
VIX = load_macro('VIX')

# ---- rebuild each library factor panel from its stored calculation ----
def build_lib_factor(name):
    M20 = C / C.shift(20) - 1.0
    M60 = C / C.shift(60) - 1.0
    M120 = C / C.shift(120) - 1.0
    M10 = C / C.shift(10) - 1.0
    V20 = tcal_roll(R, 20, 'std')
    V60 = tcal_roll(R, 60, 'std')
    if name == 'trend_r2_30_signed':
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            v = C[s].dropna()
            r2 = pd.Series(np.nan, index=v.index, dtype=float)
            x = np.arange(30)
            for i in range(29, len(v)):
                y = v.iloc[i-29:i+1].values
                if np.all(np.isfinite(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    yhat = np.polyval(np.polyfit(x, y, 1), x)
                    ss_res = np.sum((y - yhat) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2v = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    r2.iloc[i] = r2v * np.sign(slope)
            out[s] = r2.reindex(C.index)
        return out
    if name == 'semi_down_ratio_20':
        neg = R.where(R < 0, 0.0)
        return tcal_roll(neg, 20, 'std') / V20
    if name == 'mom_120d_skip5':
        return C.shift(5) / C.shift(125) - 1.0
    if name == 'mom_10d_skip5':
        return C.shift(5) / C.shift(15) - 1.0
    if name == 'time_under_water_120':
        hi = tcal_roll(C, 120, 'max')
        dd = (C / hi - 1.0).replace([np.inf, -np.inf], np.nan)
        return tcal_roll(dd, 120, 'min')
    if name == 'vol_of_vol20x60':
        return tcal_roll(V20, 60, 'std') / V20
    if name == 'dxy_beta_60':
        dm = DXY.pct_change()
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            a = pd.concat([R[s], dm], axis=1).dropna()
            if len(a) > 60:
                b = a.rolling(60).cov().iloc[:, 1].unstack() if False else None
            # simple rolling beta via loop
            vals = a.rolling(60).cov().dropna()
            # rolling cov between ret and dxy using cov method
            rcov = R[s].rolling(60).cov(dm)
            rvar = dm.rolling(60).var()
            out[s] = (rcov / rvar).reindex(C.index)
        return out
    if name == 'WTI_BETA_60':
        wm = R['WTI']
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            rcov = R[s].rolling(60).cov(wm)
            rvar = wm.rolling(60).var()
            out[s] = (rcov / rvar).reindex(C.index)
        return out
    if name == 'vix_beta_cond_60x20':
        vm = VIX.pct_change()
        vix_hi = VIX > VIX.rolling(60).median()
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            base = R[s].rolling(60).cov(vm) / vm.rolling(60).var()
            cond = pd.Series(np.nan, index=C.index, dtype=float)
            # beta only on high-vix days using rolling 60d window ending at t
            for i in range(59, len(C)):
                idx = C.index[max(0, i-59):i+1]
                sub = pd.concat([R[s], vm], axis=1).loc[idx]
                sub = sub[vix_hi.loc[idx]]
                if len(sub) >= 20:
                    c = np.cov(sub.iloc[:, 0], sub.iloc[:, 1])
                    if c[1, 1] > 0:
                        cond.iloc[i] = c[0, 1] / c[1, 1]
            out[s] = cond
        return out
    if name == 'tail_ratio_20':
        q95 = tcal_roll(R, 20, 'quantile', 0.95) if False else None
        up = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        dn = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for s in C.columns:
            v = R[s].dropna()
            up[s] = v.rolling(20).quantile(0.95).reindex(C.index)
            dn[s] = v.rolling(20).quantile(0.05).reindex(C.index)
        return (up.abs() / dn.abs()).replace([np.inf, -np.inf], np.nan)
    if name == 'kurt_20':
        return tcal_roll(R, 20, 'kurt')
    return None

print('\n%-22s %8s %8s %6s %6s %7s %6s %8s %8s %s' % (
    'factor', 'IC', 'ICIR', 'hit', 'n', 'covA', 'covD', 'turn', 'maxrho', 'stored_IC/stored_ICIR'))
for fid in lib_factors:
    d = json.load(open('factors/%s.json' % fid))
    m = d.get('validation', {}).get('metrics', {})
    fp = build_lib_factor(fid)
    if fp is None:
        print('%-22s BUILD FAILED' % fid)
        continue
    try:
        summ = L.full_validate(fp, R, horizon=10, label=fid)
    except Exception as e:
        print('%-22s VALIDATION ERROR: %s' % (fid, e))
        continue
    stored = (m.get('ic'), m.get('icir'))
    gate_ic = abs(summ['ic']) >= 0.007
    gate_icir = abs(summ['icir']) >= 0.084
    flag = 'PASS' if (gate_ic and gate_icir) else 'FAIL'
    print('%-22s %8.4f %8.4f %6.3f %6d %7.3f %6.3f %8.3f %8.3f  stored=(%.4f, %.4f) [%s]' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        summ['coverage_asset_days'], summ['coverage_dates_ge8'], summ['turnover_10d_rank'],
        summ['max_abs_library_correlation'], stored[0], stored[1], flag))
