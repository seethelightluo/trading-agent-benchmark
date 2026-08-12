import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={};H={};L={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:
  q=x.set_index('date');D[s]=q.close.astype(float);H[s]=q.high.astype(float);L[s]=q.low.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); hi=pd.DataFrame(H).reindex(p.index).ffill();lo=pd.DataFrame(L).reindex(p.index).ffill();r=p.pct_change()
# Close-location pressure: repeated closes near the low, weighted by realized range shock,
# is a transparent exhaustion/reversal signal. Cross-sectionally rank-normalized.
rg=(hi-lo)/p.replace(0,np.nan); clv=((p-lo)/(hi-lo).replace(0,np.nan)-.5)
pressure=(clv*rg).rolling(3,min_periods=2).mean()
vol=r.rolling(20,min_periods=15).std()
f=(-pressure/vol).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');x=a.ic
print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 q=a.loc[mask].ic; print(nm,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1);q=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC',np.nanmean(q),'n',len(q))
f.to_csv('scripts/miner_3_20300808_clv_pressure_signal.csv')
