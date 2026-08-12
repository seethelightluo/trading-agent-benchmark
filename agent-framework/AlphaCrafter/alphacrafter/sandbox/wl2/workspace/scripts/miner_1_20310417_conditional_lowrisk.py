import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100:d=get_index_daily_data(s,days=3000)
 if d is not None:D[s]=d.set_index('date')
p=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); r=p.pct_change()
# Conditional low-risk persistence: favor assets with low recent volatility,
# but only when their 20d return is not materially negative. This is an
# interpretable defensive quality signal for the 10-day allocation horizon.
vol=r.rolling(20,min_periods=12).std(); mom=r.rolling(20,min_periods=12).sum()
f=-vol/(r.rolling(60,min_periods=30).std()+1e-8)
f=f.where(mom.ge(mom.median(axis=1),axis=0), -0.5*vol/(r.rolling(60,min_periods=30).std()+1e-8))
f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-31',a.index>='2026-01-01')]:
 z=a.loc[m].ic;print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [3,5,10]:
 yy=p.pct_change(h).shift(-h)/h; vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),yy.iloc[i].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:vals.append(z.f.corr(z.y))
 print('decay',h,round(np.nanmean(vals),6),len(vals))
f.to_csv('scripts/miner_1_20310417_conditional_lowrisk_signal.csv')
