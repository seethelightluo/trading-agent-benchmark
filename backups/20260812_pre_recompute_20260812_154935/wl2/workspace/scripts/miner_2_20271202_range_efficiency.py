import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,5000)
    if d is None or len(d)<300: d=get_index_daily_data(s,5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
px=pd.concat(frames,axis=1).sort_index().loc['2020-01-01':]
# Range-efficiency persistence: causal lag of 10d net move / path length.
r=np.log(px).diff()
net=np.log(px/px.shift(10))
path=r.abs().rolling(10,min_periods=8).sum()
f=(net/path).shift(1)
# cross-sectional ranks/median demean, calculate IC to forward returns
f=f.sub(f.median(axis=1),axis=0)
rets=px.shift(-1)/px-1
rows=[]
for h in [1,3,5,10]:
    fr=px.shift(-h)/px-1
    vals=[]
    ns=[]
    for dt in f.index:
        a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
    z=pd.Series(vals).dropna()
    ic=z.mean(); ir=ic/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan
    print(f'h{h}: dates={len(z)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ir:.6f} hit={(z>0).mean():.4f}')
# diagnostics date coverage and rank turnover
valid=f.notna().sum(axis=1); print('assets',len(px.columns),'rows',len(px),'coverage',valid.mean()/len(U),'avgN',valid.mean())
ranks=f.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean(); print('turnover',turn)
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 z=[]
 for dt in f.loc[a:b].index:
  q=pd.concat([f.loc[dt],rets.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 z=pd.Series(z).dropna(); print(a+'-'+b,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
print('last_date',px.index[-1])
