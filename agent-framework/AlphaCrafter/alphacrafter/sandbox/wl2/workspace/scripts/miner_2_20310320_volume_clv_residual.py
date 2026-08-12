import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None: D[s]=x.set_index('date')
cl=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index().ffill(); r=cl.pct_change()
hi=pd.DataFrame({s:x.high.astype(float) for s,x in D.items()}).reindex(cl.index).ffill(); lo=pd.DataFrame({s:x.low.astype(float) for s,x in D.items()}).reindex(cl.index).ffill()
vo=pd.DataFrame({s:x.volume.astype(float) for s,x in D.items()}).reindex(cl.index).ffill()
# Residual 3-day reversal, scaled by volatility, confirmed by abnormal volume and
# close-location extremity. All inputs are available at the decision day's close.
raw=r.rolling(3,min_periods=3).sum(); res=raw.sub(raw.median(axis=1),axis=0)
vol=r.rolling(30,min_periods=20).std(); abnormal=(vo/(vo.rolling(30,min_periods=15).median()+1e-12)).clip(0.25,4)
clv=((cl-lo)/(hi-lo+1e-12)-.5).rolling(3,min_periods=3).mean()
f=(-res/(vol+1e-8))*(1+0.35*abnormal.rank(axis=1,pct=True))*(1-0.7*clv*np.sign(res))
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
f.to_csv('scripts/miner_2_20310320_volume_clv_residual_signal.csv')
