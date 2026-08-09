import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
    try:
        x=get_stock_daily_data(a,days=4000)
        if x is not None and len(x): D[a]=x.set_index('date').close.astype(float)
    except Exception: pass
p=pd.concat(D,axis=1,sort=True).ffill(); r=p.pct_change();
# Candidate: volatility-conditioned 3d reversal, damped in high-volatility conditions.
rev=-(p/p.shift(3)-1)
vol=r.rolling(20,min_periods=10).std()
# cross-sectional volatility-neutralized reversal: divide by own recent vol, preserving reversal direction
f=rev/(vol*np.sqrt(3))
print('assets',len(D),'dates',len(p),'avgN',round(f.notna().sum(axis=1).mean(),2))
for h in [1,5,10]:
    ic=[]
    for i,dt in enumerate(p.index):
        if i+h>=len(p): continue
        q=pd.concat([f.loc[dt],(p.iloc[i+h]/p.iloc[i]-1).rename('fwd')],axis=1).dropna()
        if len(q)>=8:
            v=q.iloc[:,0].corr(q.fwd)
            if np.isfinite(v): ic.append(v)
    s=pd.Series(ic)
    print('h',h,'n',len(s),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4))
print('turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5),'coverage',round(f.notna().sum(axis=1).mean()/len(D),5))
# split regimes by median cross-asset trailing vol, with explicit date counts
rv=r.mean(axis=1).rolling(20).std()
for label,mask in [('lowvol',rv<=rv.rolling(252,min_periods=60).median()),('highvol',rv>rv.rolling(252,min_periods=60).median())]:
    vals=[]
    for i,dt in enumerate(p.index[:-1]):
        if not mask.loc[dt]: continue
        q=pd.concat([f.loc[dt],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
        if len(q)>=8: vals.append(q.iloc[:,0].corr(q.y))
    s=pd.Series(vals); print(label,'dates',len(s),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5) if len(s)>1 else np.nan)
for name,z in [('short3',rev),('short5',-(p/p.shift(5)-1)),('ram20',(p/p.shift(20)-1)/(r.rolling(60).std()*np.sqrt(20)))]:
    q=pd.concat([f.stack().rename('f'),z.stack().rename('z')],axis=1).dropna(); print('corr',name,round(q.f.corr(q.z),5),'n',len(q))
