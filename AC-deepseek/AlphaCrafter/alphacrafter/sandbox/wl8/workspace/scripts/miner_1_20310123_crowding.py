import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import miner_1_20280504_common as C

px = C.load_prices()
ret = px.pct_change()
H = 10
fwd = px.shift(-H) / px - 1.0

def cross_ic(fac, fwdv):
    ics = []
    idx = fac.index.intersection(fwdv.index)
    for d in idx:
        a, b = fac.loc[d], fwdv.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 8:
            ic = a[m].corr(b[m], method='spearman')
            if pd.notna(ic):
                ics.append((d, ic))
    return pd.Series(dict(ics)).sort_index()

def factor_corr(a, b):
    # mean daily spearman corr between two factor panels
    cs = []
    for d in a.index.intersection(b.index):
        ra, rb = a.loc[d], b.loc[d]
        m = ra.notna() & rb.notna()
        if m.sum() >= 8:
            c = ra[m].corr(rb[m], method='spearman')
            if pd.notna(c):
                cs.append(c)
    return np.mean(cs), len(cs)

def mom20_skip5(): return px.shift(5) / px.shift(25) - 1.0
def mom10_skip5(): return px.shift(5) / px.shift(15) - 1.0
def mom60_skip5(): return px.shift(5) / px.shift(65) - 1.0
def mom_accel(): return mom10_skip5() - mom60_skip5()

def rolling_beta(sr, br, w=60):
    num = sr.rolling(w).cov(br)
    dem = br.rolling(w).var()
    return num / dem.replace(0, np.nan)

def beta60_cand(base):
    br = ret[base]
    out = {}
    for s in list(ret.columns):
        out[s] = 1.0 if s == base else rolling_beta(ret[s], br, 60)
    return pd.DataFrame(out, index=ret.index)

flip_mom = np.sign(px.shift(10)/px - 1.0) * (px.shift(20)/px - 1.0)
usdcny = C.load_macro()['USDCNY'].reindex(ret.index).ffill()
usdcny_ret = usdcny.pct_change()
usdcny_beta = pd.DataFrame({s: rolling_beta(ret[s], usdcny_ret, 60) for s in list(ret.columns)}, index=ret.index)

mom_accel_f = mom_accel().reindex(px.index)
gold_beta = beta60_cand('XAU').reindex(px.index)

print('ACTIVE refs:')
for name, f in [('flip_mom_20x10', flip_mom), ('usdcny_beta_60', usdcny_beta)]:
    ic = cross_ic(f, fwd); icm = ic.mean(); icir = icm/ic.std()*np.sqrt(len(ic))
    ric = ic[ic.index>='2030-01-01'].mean()
    print(f'  {name:16s} n={len(ic):4d} IC={icm:+.4f} ICIR={icir:+.3f} recentIC(>=2030)={ric:+.4f}')

print('\nCANDIDATE mom_accel_10x60_skip5:')
ic = cross_ic(mom_accel_f, fwd); icm=ic.mean(); icir=icm/ic.std()*np.sqrt(len(ic))
ric = ic[ic.index>='2030-01-01'].mean()
print(f'  n={len(ic)} IC={icm:+.4f} ICIR={icir:+.3f} recentIC(>=2030)={ric:+.4f}')
for ref in ['flip_mom_20x10','usdcny_beta_60']:
    c,n = factor_corr(mom_accel_f, flip_mom if ref=='flip_mom_20x10' else usdcny_beta)
    print(f'  daily-rank corr vs {ref}: {c:+.3f} (n={n})')

# max abs library corr among candidates being considered together (mom_accel vs gold_beta)
c,n = factor_corr(mom_accel_f, gold_beta)
print(f'\nmom_accel corr with gold_beta_60: {c:+.3f} (n={n})')

# decay analysis for mom_accel across horizons
print('\nDecay analysis mom_accel:')
for h in [5,10,15,20]:
    fh = px.shift(-h)/px - 1.0
    ics = cross_ic(mom_accel_f, fh)
    icm = ics.mean(); icir = icm/ics.std()*np.sqrt(len(ics)) if ics.std()>0 else 0
    print(f'  H={h:2d} n={len(ics):4d} IC={icm:+.4f} ICIR={icir:+.3f}')

# coverage / turnover
n = mom_accel_f.notna().sum(axis=1)
print('\ncoverage dates>=8: %.3f' % (n>=8).mean())
r = mom_accel_f.rank(axis=1)
stab = []
for d in r.index:
    pr = r.shift(1).loc[d]; row = r.loc[d]
    m = row.notna() & pr.notna()
    if m.sum()>=8:
        c2 = row[m].corr(pr[m], method='spearman')
        if pd.notna(c2): stab.append(c2)
print('turnover(1-rankstab): %.3f' % (1-np.mean(stab)))