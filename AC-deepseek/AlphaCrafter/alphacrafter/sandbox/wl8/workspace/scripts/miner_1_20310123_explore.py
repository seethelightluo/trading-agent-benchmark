import sys, inspect
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd

import miner_1_20280504_common as C

# print helper signatures for reference
for f in ['load_prices', 'load_macro', 'cross_sectional_ic', 'ic_stats']:
    fn = getattr(C, f, None)
    if fn is not None:
        try:
            print('SIG', f, inspect.signature(fn))
        except Exception:
            print('SIG', f, 'n/a')

px = C.load_prices()
print('panel rows, cols:', px.shape, 'last date:', px.index.max())

ret = px.pct_change()
H = 10
fwd = px.shift(-H) / px - 1.0

WATCH = list(px.columns)

def cross_sectional_spearman(fac, fwdv):
    """Spearman IC per date across the cross-section (>=8 valid instruments)."""
    idx = fac.index.intersection(fwdv.index)
    ics = []
    for d in idx:
        a = fac.loc[d]
        b = fwdv.loc[d]
        mask = a.notna() & b.notna()
        if mask.sum() >= 8:
            ic = a[mask].corr(b[mask], method='spearman')
            if pd.notna(ic):
                ics.append((d, ic))
    s = pd.Series(dict(ics)).sort_index()
    return s

def report(name, fac, n_dates=None):
    n = fac.notna().sum(axis=1)
    cov = (n >= 8).mean()
    ic = cross_sectional_spearman(fac, fwd)
    if len(ic) == 0:
        print(f'{name:22s} no IC dates'); return
    icm = ic.mean()
    icir = ic.mean() / ic.std() * np.sqrt(len(ic))
    # turnover: day-over-day rank stability, ~1-avg spearman of ranks
    r = fac.rank(axis=1)
    stab = []
    dr = r.diff()
    for d in r.index:
        prev = r.shift(1).loc[d]
        row = r.loc[d]
        m = row.notna() & prev.notna()
        if m.sum() >= 8:
            c = row[m].corr(prev[m], method='spearman')
            if pd.notna(c):
                stab.append(c)
    stabm = np.mean(stab) if stab else np.nan
    turnover = 1 - stabm if pd.notna(stabm) else np.nan
    recent = ic[ic.index >= '2030-01-01']
    ric = recent.mean() if len(recent) else np.nan
    print(f'{name:22s} n_dates={len(ic):4d} coshare={cov:.2f} IC={icm:+.4f} '
          f'ICIR={icir:+.3f} turn={turnover:.3f} recentIC(>=2030)={ric:+.4f}')

def zscore(df):
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return (df.sub(m, axis=0)).div(s.replace(0, np.nan), axis=0)

# ---- candidate factor definitions ----
def mom20_skip5():
    return px.shift(5) / px.shift(25) - 1.0

def mom10_skip5():
    return px.shift(5) / px.shift(15) - 1.0

def mom60_skip5():
    return px.shift(5) / px.shift(65) - 1.0

def momentum_accel():
    return mom10_skip5() - mom60_skip5()

def vol_adj_mom():
    rv = ret.rolling(20).std()
    return mom10_skip5() / rv.replace(0, np.nan)

def low_vol():
    return -ret.rolling(60).std()

def reversal_5():
    return -ret.rolling(5).sum()

def drawdown_60():
    return px / px.rolling(60).max() - 1.0

def beta60(base):
    rv = ret[list(px.columns)]
    base_r = ret[base].rename('b')
    cov = rv.rolling(60).cov(base_r).groupby(level=0).mean()  # not used
    # covariance of each sym with base over 60d
    b = base_r
    cb = rv.rolling(60).corr(b).mul(rv.rolling(60).std().div(b.rolling(60).std()) if False else 1)
    return cb

# robust rolling beta implementation (symmetric window)
def rolling_beta(sym_ret, base_ret, w=60):
    # align
    df = pd.concat([sym_ret, base_ret], axis=1, keys=['s', 'b'])
    df['s'] = sym_ret
    df['b'] = base_ret
    num = sym_ret.rolling(w).cov(base_ret)
    dem = base_ret.rolling(w).var()
    return num / dem.replace(0, np.nan)

def beta60_cand(base):
    br = ret[base]
    out = {}
    for s in list(ret.columns):
        if s == base:
            out[s] = 1.0
        else:
            out[s] = rolling_beta(ret[s], br, 60)
    return pd.DataFrame(out, index=ret.index)

print('\n--- Explored candidates (H=10, full period) ---')
cands = {
    'mom20_skip5': mom20_skip5(),
    'mom10_skip5': mom10_skip5(),
    'mom_accel_10x60_skip5': momentum_accel(),
    'vol_adj_mom10_skip5': vol_adj_mom(),
    'low_vol_60': low_vol(),
    'reversal_5': reversal_5(),
    'drawdown_60': drawdown_60(),
    'gold_beta_60': beta60_cand('XAU'),
    'oil_beta_60': beta60_cand('WTI'),
    'usdcny_beta_60': None,
}
if hasattr(C, 'load_macro'):
    try:
        macro = C.load_macro()
        print('\nmacro columns:', list(macro.columns) if hasattr(macro,'columns') else type(macro))
        if hasattr(macro, 'columns') and 'USDCNY' in macro.columns:
            u = macro['USDCNY'].reindex(ret.index).ffill()
            usd_ret = u.pct_change()
            cands['usdcny_beta_60'] = beta60_cand_us = None
            # build usdcny beta
            ob = {}
            for s in list(ret.columns):
                ob[s] = rolling_beta(ret[s], usd_ret, 60)
            cands['usdcny_beta_60'] = pd.DataFrame(ob, index=ret.index).reindex(cands['mom20_skip5'].index)
    except Exception as e:
        print('macro err', e)

for name, f in cands.items():
    if f is None:
        continue
    report(name, f)

# ---- crowding vs active flip_mom and usdcny_beta ----
flip_mom = np.sign(px.shift(10)/px - 1.0) * (px.shift(20)/px - 1.0)
print('\n--- Active ref factor (flip_mom_20x10) ---')
report('flip_mom_20x10', flip_mom)

print('\nDone.')