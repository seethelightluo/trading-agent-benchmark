import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None: D[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Acceleration: recent 5d return minus average daily 20d trend, scaled by recent risk.
# Cross-sectional demean avoids market direction; completed-bar signal predicts next day.
a5=p.pct_change(5); a20=p.pct_change(20)/4
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(a5-a20).sub((a5-a20).median(axis=1),axis=0).div(vol).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6),'coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for k in [3,5,10]:
 y=r.rolling(k).sum().shift(-k+1); o=[]
 for i in range(len(p)-k):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',round(np.nanmean(o),6),'n',len(o))
f.to_csv('scripts/miner_2_20301226_acceleration_signal.csv')
