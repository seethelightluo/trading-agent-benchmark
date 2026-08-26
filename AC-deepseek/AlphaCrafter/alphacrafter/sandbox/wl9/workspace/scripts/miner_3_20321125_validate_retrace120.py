"""miner_3 2032-11-25: in-depth validation of candidate retrace_high120.
Visible through 2032-11-24. No lookahead. 15-asset cross-asset universe.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END='2032-11-24'
STOCK_DIR=Path('../persistent/stock_data'); INDEX_DIR=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
        'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets,end):
    out={}
    for a in assets:
        f=STOCK_DIR/f'{a}.csv'
        if not f.exists(): f=INDEX_DIR/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date'])
        out[a]=df[df['date']<=end].sort_values('date').set_index('date')['close'].astype(float)
    return out

closes=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(closes).dropna()
rets=close.pct_change().dropna()
fwd5=rets.shift(-5).rolling(5).mean()
fwd10=rets.shift(-10).rolling(10).mean()
fwd20=rets.shift(-20).rolling(20).mean()
print(f"Panel {close.shape[0]}x{close.shape[1]} {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}",flush=True)

cand=(close/close.rolling(120).max()-1).reindex(fwd10.index)

def compute_ic(fv,fwd,start=None,min_dates=30):
    fv=fv.reindex(fwd.index); idx=fwd.index
    if start: idx=idx[idx>=pd.Timestamp(start)]
    ics=[]
    for d in idx:
        f=fv.loc[d]; r=fwd.loc[d]; m=f.notna()&r.notna()
        if m.sum()>=8:
            a=f[m].rank().values; b=r[m].rank().values
            if a.std()>0 and b.std()>0: ics.append(np.corrcoef(a,b)[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(ic=0.0,icir=0.0,n=len(ics),hit=0.0)
    mu=ics.mean(); sd=ics.std(); icir=mu/sd*np.sqrt(len(ics)) if sd>0 else 0
    return dict(ic=float(mu),icir=float(icir),n=len(ics),hit=float((ics>0).mean()))

print("\n== decay (full sample, IC=ICIR using respective horizon) ==")
for h,fd in [('5',fwd5),('10',fwd10),('20',fwd20)]:
    r=compute_ic(cand,fd); print(f"  {h}d: IC={r['ic']:.4f} ICIR={r['icir']:.4f} n={r['n']} hit={r['hit']:.3f}")

print("\n== sub-regime IC (10d horizon) ==")
for s in ['2022-01-01','2024-01-01','2026-01-01','2028-01-01','2030-01-01','2031-06-01','2032-01-01','2032-06-01']:
    r=compute_ic(cand,fwd10,start=s)
    print(f"  from {s}: IC={r['ic']:.4f} ICIR={r['icir']:.4f} n={r['n']} hit={r['hit']:.3f}")

cov=float(cand.notna().mean().mean())
def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())
print(f"coverage={cov:.3f} turnover={turnover(cand):.3f}")