import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:D[s]=x.set_index('date')
cl=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index().ffill(); r=cl.pct_change()
hi=pd.DataFrame({s:x.high.astype(float) for s,x in D.items()}).reindex(cl.index).ffill(); lo=pd.DataFrame({s:x.low.astype(float) for s,x in D.items()}).reindex(cl.index).ffill()
vol=pd.DataFrame({s:x.volume.astype(float) for s,x in D.items()}).reindex(cl.index).ffill()
# Volume-confirmed residual reversal: reverse 3-day relative move, but emphasize abnormal-volume exhaustion.
res=r.rolling(3,min_periods=3).sum().sub(r.median(axis=1).rolling(3,min_periods=3).sum(),axis=0)
rv=r.rolling(30,min_periods=20).std(); vs=(vol/(vol.rolling(30,min_periods=20).median()+1e-12)).clip(0,5)
clv=((cl-lo)/(hi-lo+1e-12)-.5).abs()
f=(-res/(rv+1e-8))*(1+0.5*vs.rank(axis=1,pct=True))*(1+0.35*clv)
rows=[]
for i in range(len(cl)-10):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((cl.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-31',a.index>='2026-01-01')]:
 z=a.loc[mask].ic; print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [1,3,5,10]:
 rr=r.rolling(h).sum().shift(-h+1); vals=[]
 for i in range(len(cl)-h):
  z=pd.concat([f.iloc[i].rename('f'),rr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append(z.f.corr(z.y))
 print('decay',h,round(np.mean(vals),6),len(vals))
f.to_csv('scripts/miner_2_20310220_volume_clv_exhaustion_signal.csv')
