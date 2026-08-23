import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=get_stock_daily_data(s,2500)
 if d is None or len(d)==0:d=get_index_daily_data(s,2500)
 return d.set_index(pd.to_datetime(d.date)).close
p=pd.concat({s:ld(s) for s in U},axis=1).sort_index().ffill(); r=p.pct_change(); b=r['SPX']
# rolling beta, computed explicitly per column for stable pandas compatibility
beta=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in U:
 beta[s]=r[s].rolling(60,min_periods=40).cov(b)/b.rolling(60,min_periods=40).var()
raw=r.rolling(20,min_periods=15).sum(); br=b.rolling(20,min_periods=15).sum()
f=(raw-beta.shift(1).mul(br,axis=0)).shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));dates.append(d);ns.append(len(a))
 z=pd.Series(vals,index=dates).dropna();print('H',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 if h==1:
  print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'period',z.index.min().date(),z.index.max().date())
  for nm,a,bx in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31')]:
   q=z.loc[a:bx];print('REG',nm,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),4))
print('candidate=residual_momentum')
