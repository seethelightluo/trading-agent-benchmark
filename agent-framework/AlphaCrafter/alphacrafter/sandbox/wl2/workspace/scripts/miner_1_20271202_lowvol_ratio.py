import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in u:
 x=get_stock_daily_data(s,days=2200)
 if x is None or len(x)<80:x=get_index_daily_data(s,days=2200)
 if x is not None and len(x):D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();
# Low-volatility defensive factor, smoothed by 5d change in realized risk.
v20=r.rolling(20).std(); v60=r.rolling(60).std(); f=-(v20/v60)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic.dropna();print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean());print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-27',a.index>='2026-01-01')]:
 q=a.loc[mask,'ic'].dropna();print(name,len(q),q.mean(),q.mean()/q.std(ddof=1))
