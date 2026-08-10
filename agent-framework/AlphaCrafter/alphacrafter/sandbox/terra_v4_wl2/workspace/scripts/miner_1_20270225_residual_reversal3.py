import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change();
# residual short-term reversal: negative 3d return relative to same-day cross-sectional median
raw=r.rolling(3).sum().shift(1); f=-(raw.sub(raw.median(axis=1),axis=0))
fr=p.pct_change(1).shift(-1); ic=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
a=np.array(ic);a=a[np.isfinite(a)]
print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for yr,g in f.groupby(f.index.year):
 q=[]
 for dt in g.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q);q=q[np.isfinite(q)]
 print('Y',yr,'n',len(q),'ic',round(q.mean(),5),'icir',round(q.mean()/q.std(ddof=1),5))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_residual_reversal3.csv',index=False)
