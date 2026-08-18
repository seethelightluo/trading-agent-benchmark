import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=5000)
    except Exception: x=None
    if x is None:
        try: x=get_index_daily_data(s, days=5000)
        except Exception: x=None
    if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change(); bench=r.mean(axis=1)
asset40=p.pct_change(40); b40=bench.rolling(40).sum(); vol20=r.rolling(20).std()*np.sqrt(252)
raw=asset40.sub(b40,axis=0); f=raw.div(vol20.replace(0,np.nan)).where(b40>0).shift(1)
for h in [5,10,20,40]:
    fw=p.shift(-h)/p-1; ics=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    q=pd.Series(ics).dropna()
    print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('assets',len(p.columns),'rows',len(p),'coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'active',round((f.notna().sum(axis=1)>0).mean(),4)); print('period',p.index.min(),p.index.max())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_2_20350525_relative_strength_signal.csv',index=False)
