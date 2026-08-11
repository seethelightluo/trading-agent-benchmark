import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in u:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<80:x=get_index_daily_data(s,days=2200)
 if x is not None and len(x):D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();v=r.rolling(20,min_periods=10).std(); d=r.rolling(5,min_periods=3).std().mean(axis=1); med=d.rolling(60,min_periods=20).median()
# Continuous cross-asset dispersion percentile, avoiding a binary regime boundary.
w=(d/(med+1e-8)).clip(.5,2.0)-.5; w=w.clip(0,1)
f=-r.rolling(3,min_periods=3).sum()/v; f=f.mul(.15+.85*w,axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  z.f=z.f.clip(z.f.quantile(.05),z.f.quantile(.95));rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic.dropna();print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean());print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 q=a.loc[mask].ic.dropna();print(name,len(q),q.mean(),q.mean()/q.std(ddof=1))
