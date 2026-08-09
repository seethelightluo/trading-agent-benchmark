import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x): return x
        except Exception: pass
    return None
D={s:load(s) for s in U}; px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index()
r=px.pct_change(); n=r.notna().sum(axis=1)
# Continuous breadth intensity: reversal magnitude is increased smoothly as breadth
# of prior-day losers departs from neutral, avoiding sparse hard threshold activation.
breadth=(r.lt(0).sum(axis=1)/n).shift(1)
base=-r.rolling(3).sum().shift(1) # signal known at t from returns through t-1
# centered signed stress intensity; positive means more losers, with symmetric response
intensity=(breadth-0.5).abs()*2
f=base.mul(intensity,axis=0)
f=f.sub(f.median(axis=1),axis=0)
fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
rows=[]
for h,x in fr.items():
    vals=[]; ns=[]; dates=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
    a=np.asarray(vals); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(f.notna().sum().sum()/(len(U)*len(f)),4))
    for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-12-31')]:
        q=np.array([v for v,d in zip(vals,dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)])
        if len(q)>1: print(' regime',lo[:4],len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
    if h==1:
        out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_2_20270225_continuous_breadth_reversal3.csv',index=False)
