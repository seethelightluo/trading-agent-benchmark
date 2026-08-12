import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None:
  d=d.set_index('date'); D[s]=d[['open','close']].astype(float)
o=pd.DataFrame({s:d.open for s,d in D.items()}).sort_index().ffill(); c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill();
# Fade three-day average overnight gap, normalized by trailing 20d close volatility.
gap=(o/c.shift(1)-1).replace([np.inf,-np.inf],np.nan)
ret=c.pct_change(); vol=ret.rolling(20,min_periods=10).std()
signal=-(gap.rolling(3,min_periods=3).mean()/vol)
signal=signal.sub(signal.median(axis=1),axis=0)
rows=[]
for i in range(len(ret)-1):
 z=pd.concat([signal.iloc[i].rename('f'),ret.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((ret.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',signal.notna().mean().mean(),'turnover',signal.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for k in [3,5,10]:
 y=ret.rolling(k).sum().shift(-k+1); vals=[]
 for i in range(len(ret)-k):
  z=pd.concat([signal.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',np.nanmean(vals),'n',len(vals))
signal.to_csv('scripts/miner_1_20301128_gap_reversal_signal.csv')
