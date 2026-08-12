import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<100:d=get_index_daily_data(s,3000)
 if d is not None:D[s]=d.set_index('date')
cl=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill();r=cl.pct_change()
# Downside-risk quality: prefer assets whose recent downside deviation is low
# relative to total volatility, neutralized cross-sectionally.
down=r.clip(upper=0)
ratio=down.rolling(20,min_periods=12).std()/(r.rolling(60,min_periods=30).std()+1e-8)
f=-ratio
f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i in range(len(cl)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:rows.append((cl.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-31',a.index>='2026-01-01')]:
 z=a.loc[m].ic;print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [1,3,5,10]:
 rr=r.rolling(h).sum().shift(-h+1);v=[]
 for i in range(len(cl)-h):
  z=pd.concat([f.iloc[i].rename('f'),rr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:v.append(z.f.corr(z.y))
 print('decay',h,round(np.nanmean(v),6),len(v))
f.to_csv('scripts/miner_2_20310403_downside_quality_signal.csv')
