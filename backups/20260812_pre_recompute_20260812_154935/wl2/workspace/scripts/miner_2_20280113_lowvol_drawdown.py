import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in u:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=2200)
 if x is not None and len(x):D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change();
# Volatility-managed defensive rank: lower realized volatility and lower recent drawdown receive higher score.
vol=r.rolling(20,min_periods=12).std(); dd=p/p.rolling(60,min_periods=30).max()-1
f=-(0.7*vol + 0.3*(-dd))
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  q=z.f.clip(z.f.quantile(.05),z.f.quantile(.95));rows.append((p.index[i],len(z),q.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic.dropna()
print('dates',len(x),'avgN',round(a.n.mean(),2),'IC',round(x.mean(),8),'ICIR',round(x.mean()/x.std(ddof=1),8),'hit',round((x>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'coverage',round(f.notna().mean().mean(),6),'period',a.index.min(),a.index.max(),'instruments',len(D))
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 q=a.loc[mask].ic.dropna();print(name,len(q),round(q.mean(),8),round(q.mean()/q.std(ddof=1),8))
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1);rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:rr.append(z.f.clip(z.f.quantile(.05),z.f.quantile(.95)).corr(z.y))
 print('h',h,'IC',round(float(np.nanmean(rr)),8),'n',len(rr))
