"""
Complete Factor Re-validation & Exploration
Date: 2032-05-13 (visible through 2032-05-12)

Re-validate: flip_mom_20x10, mom_diff_20_60
Explore new: vol_adjusted_mom_20, zscore_20, csi_mom_40
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd, numpy as np
from scipy.stats import pearsonr
import json, os

np.random.seed(42)

watchlist = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

print("Fetching data...")
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=750)
    if df is not None and len(df) > 200:
        data[sym] = df
    else:
        print(f"  WARN: {sym} {len(df) if df is not None else 0}")
print(f"Loaded {len(data)} instruments")

vix = get_index_daily_data(symbol='VIX', days=750)
dxy = get_index_daily_data(symbol='DXY', days=750)
print(f"VIX={vix is not None}, DXY={dxy is not None}")

# FACTOR DEFINITIONS
def flip_mom(df, lb=20):
    c = df['close'].values
    if len(c) < lb+2: return None
    rm = pd.Series(c).rolling(lb).min().values
    pm = np.roll(rm, 1); pm[0]=np.nan
    r = 1 - c/pm
    return np.where(np.isfinite(r), r, np.nan)

def mom_diff(df):
    c = df['close'].values; n=len(c)
    if n<61: return None
    m20,m60 = np.full(n,np.nan),np.full(n,np.nan)
    if n>20: m20[20:]=c[20:]/c[:-20]-1
    if n>60: m60[60:]=c[60:]/c[:-60]-1
    return m20-m60

def vol_adj_mom(df, lm=20, lvs=10, lvl=60):
    c=df['close'].values; n=len(c)
    if n<max(lm,lvl)+2: return None
    mom=np.full(n,np.nan)
    if n>lm: mom[lm:]=c[lm:]/c[:-lm]-1
    r=np.full(n,np.nan); r[1:]=c[1:]/c[:-1]-1
    sv=pd.Series(np.abs(r)).rolling(lvs).mean().values
    lv=pd.Series(np.abs(r)).rolling(lvl).mean().values
    vr=np.full(n,np.nan); m=lv>1e-10; vr[m]=sv[m]/lv[m]
    sc=np.clip(1.5-0.5*vr,0.3,1.7)
    return np.where(np.isfinite(mom*sc), mom*sc, np.nan)

def zscore_20(df):
    c=df['close'].values; n=len(c)
    if n<21: return None
    m=pd.Series(c).rolling(20).mean().values
    s=pd.Series(c).rolling(20).std().values
    z=np.full(n,np.nan); mx=s>1e-10; z[mx]=(c[mx]-m[mx])/s[mx]
    return np.where(np.isfinite(z), z, np.nan)

def csi_mom_40(df):
    """Cross-asset score: 1 - close/rolling_max(close,40) (drawdown from peak)"""
    c=df['close'].values; n=len(c)
    if n<41: return None
    rmax=pd.Series(c).rolling(40).max().values
    pmax=np.roll(rmax,1); pmax[0]=np.nan
    r=1-c/pmax
    return np.where(np.isfinite(r), r, np.nan)

# IC COMPUTATION
def ic_series(fn, fwd=10, max_dt=None):
    fs=list(data.keys())[0]
    ad=data[fs]['date'].values; n=len(ad)
    ics,ds,cs=[],[],[]
    si=150; ei=n-fwd
    if max_dt and ei-si>max_dt: si=ei-max_dt
    for i in range(si, ei):
        dt=ad[i]; fv,rv={},{}
        for sym in data:
            df=data[sym]; fa=fn(df.iloc[:i+1])
            if fa is not None and len(fa)>1:
                v=fa[-1]
                if np.isfinite(v): fv[sym]=v
        for sym in data:
            df=data[sym]
            if i+fwd<len(df):
                r=df.iloc[i+fwd]['close']/df.iloc[i]['close']-1
                if np.isfinite(r): rv[sym]=r
        com=set(fv)&set(rv)
        if len(com)>=8:
            fva=np.array([fv[s] for s in com])
            rva=np.array([rv[s] for s in com])
            mk=np.isfinite(fva)&np.isfinite(rva)
            if mk.sum()>=8:
                ic,_=pearsonr(fva[mk],rva[mk])
                ics.append(ic); ds.append(dt); cs.append(mk.sum())
    a=np.array(ics)
    if len(a)==0: return dict(ic=0,icir=0,n=0,cov=0)
    m,s=np.mean(a),np.std(a); ir=m/s if s>0 else 0
    ht=np.mean(np.sign(a)==np.sign(m)) if m!=0 else 0.5
    return dict(ic=float(f'{m:.6f}'),icir=float(f'{ir:.6f}'),
                hit=float(f'{ht:.4f}'),n=len(a),
                cov=float(f'{np.mean(cs):.1f}'),
                frm=str(ds[0])[:10],to=str(ds[-1])[:10])

# RUN ALL
print("\n"+"="*70)
print("RE-VALIDATION: flip_mom_20x10 (original)")
print("="*70)
for fwd in [5,10,21]:
    r=ic_series(flip_mom, fwd)
    print(f"  fwd{fwd:2d}d: IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['hit']:.2f} n={r['n']} cov={r['cov']:.1f} {r['frm']}->{r['to']}")

print("\n-- flip_mom_20x10 (last 300 dates, ~1yr) --")
for fwd in [5,10,21]:
    r=ic_series(flip_mom, fwd, max_dt=300)
    print(f"  fwd{fwd:2d}d: IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['hit']:.2f} n={r['n']} cov={r['cov']:.1f}")

print("\n"+"="*70)
print("RE-VALIDATION: mom_diff_20_60")
print("="*70)
for fwd in [5,10,21]:
    r=ic_series(mom_diff, fwd)
    print(f"  fwd{fwd:2d}d: IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={r['hit']:.2f} n={r['n']} cov={r['cov']:.1f} {r['frm']}->{r['to']}")

print("\n-- mom_diff_20_60 (last 300 dates) --")
for fwd in [5,10,21]:
    r=ic_series(mom_diff, fwd, max_dt=300)
    print(f"  fwd{fwd:2d}d: IC={r['ic']:.4f} ICIR={r['icir']:.4f} hit={