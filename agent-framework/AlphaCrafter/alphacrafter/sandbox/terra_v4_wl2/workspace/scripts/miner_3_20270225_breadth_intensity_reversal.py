import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x): return x
        except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); fwd={h:px.shift(-h)/px-1 for h in (1,3,5,10)}
# Lagged breadth avoids using today's close to form today's signal.
b=(-r).gt(0).sum(axis=1).div(r.notna().sum(axis=1)).shift(1)
# Continuous breadth intensity: stronger reversal only when lagged market breadth is unusually bearish.
intensity=((b-0.5).clip(lower=0)*2).rolling(20,min_periods=20).mean().shift(1)
raw=-r.rolling(3).sum().mul(intensity,axis=0)
f=raw.sub(raw.median(axis=1),axis=0)
for h,y in fwd.items():
    a=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1:
            a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    a=np.asarray(a); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',float(f.notna().sum().sum()/(len(U)*len(f))))
for name,(lo,hi) in {'2020_22':('2020-01-01','2022-12-31'),'2023_24':('2023-01-01','2024-12-31'),'2025_26':('2025-01-01','2026-12-31'),'2027':('2027-01-01','2027-02-25')}.items():
    aa=[]
    for dt in f.loc[lo:hi].index:
        z=pd.concat([f.loc[dt],fwd[1].loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    print('regime',name,'dates',len(aa),'IC',round(float(np.mean(aa)),6) if aa else None)
# save a reproducible signal artifact for audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_3_20270225_breadth_intensity_reversal.csv',index=False)
