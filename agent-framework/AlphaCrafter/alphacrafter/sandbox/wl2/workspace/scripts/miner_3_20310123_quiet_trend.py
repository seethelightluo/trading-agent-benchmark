import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Quiet-trend continuation: medium trend is favored when short volatility is compressed
# versus the 60-day baseline. All inputs are lagged at the signal date.
r20=r.rolling(20,min_periods=15).sum(); v10=r.rolling(10,min_periods=8).std(); v60=r.rolling(60,min_periods=40).std()
compression=(v60/(v10+1e-8)).clip(0.5,3.0)
f=r20.div(v60+1e-8).mul(compression)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for k in [3,5,10]:
 y=r.rolling(k).sum().shift(-k+1);o=[]
 for i in range(len(p)-k):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',round(np.nanmean(o),6),'n',len(o))
f.to_csv('scripts/miner_3_20310123_quiet_trend_signal.csv')
