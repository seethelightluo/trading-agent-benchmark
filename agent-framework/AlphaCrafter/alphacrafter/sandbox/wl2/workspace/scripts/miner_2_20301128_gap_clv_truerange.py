import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=4000)
 if d is not None: D[s]=d.set_index('date')[['open','close','high','low']].astype(float)
P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill(); R=P.pct_change()
O=pd.DataFrame({s:d.open for s,d in D.items()}).reindex(P.index).ffill()
H=pd.DataFrame({s:d.high for s,d in D.items()}).reindex(P.index).ffill(); L=pd.DataFrame({s:d.low for s,d in D.items()}).reindex(P.index).ffill()
# Reversal of a short-horizon overnight gap plus close-location shock; scale by true-range risk.
gap=O/P.shift(1)-1
clv=((P-L)-(H-P))/(H-L).replace(0,np.nan)
tr=(H-L)/P.shift(1)
shock=gap.rolling(2,min_periods=2).mean()+0.40*clv.rolling(2,min_periods=2).mean()
risk=tr.rolling(20,min_periods=15).mean()
F=(-shock/risk).sub((-shock/risk).median(axis=1),axis=0)
rows=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i].rename('f'),R.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1))
for k in [3,5,10]:
 y=R.rolling(k).sum().shift(-k+1); o=[]
 for i in range(len(P)-k):
  z=pd.concat([F.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',np.nanmean(o),'n',len(o))
F.to_csv('scripts/miner_2_20301128_gap_clv_truerange_signal.csv')
