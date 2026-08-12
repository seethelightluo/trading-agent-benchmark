import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None:
  D[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Three-day shock, scaled by trailing volatility, with cross-sectional median removed.
vol=r.rolling(20,min_periods=15).std()
shock=r.rolling(3,min_periods=3).sum()/vol.replace(0,np.nan)
resid=shock.sub(shock.median(axis=1),axis=0)
disp=r.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
f=-resid.mul(gate,axis=0)
rows=[]
for i in range(len(r)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((r.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1))
for k in [3,5,10]:
 y=r.rolling(k).sum().shift(-k+1); o=[]
 for i in range(len(r)-k):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',np.nanmean(o),'n',len(o))
f.to_csv('scripts/miner_1_20301212_shock_residual_reversal_signal.csv')
