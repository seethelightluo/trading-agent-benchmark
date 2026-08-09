import numpy as np,pandas as pd
from pathlib import Path
root=Path('../persistent'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-09-23')
def L(s,m=False):
 p=root/('index_data' if m else 'stock_data')/(s+'.csv'); return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.concat({s:L(s) for s in U},axis=1); v=L('VIX',True); p=p.join(v.rename('VIX'),how='inner').loc[:C]; v=p.pop('VIX'); r=p.pct_change(); rev=-r.rolling(5,min_periods=5).sum(); mom=r.rolling(20,min_periods=15).sum()
for quant in [.50,.60,.70,.75,.80]:
 stress=v>v.rolling(60,min_periods=30).quantile(quant); f=rev.where(stress,mom); ys=p.shift(-1).div(p)-1; a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],ys.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(a); print('q',quant,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('cutoff',C.date())
