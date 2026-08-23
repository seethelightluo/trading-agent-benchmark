import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(C).sort_index(); px.index=pd.to_datetime(px.index); r=np.log(px).diff()
csvol=r.rolling(20,min_periods=15).std().median(axis=1); stress=(csvol/csvol.rolling(60,min_periods=30).median()).clip(.5,2)
vol=r.rolling(40,min_periods=30).std()*np.sqrt(252); raw=(-r.rolling(10,min_periods=10).sum()/vol)*stress
sig=raw.sub(raw.median(axis=1),axis=0); fwd=np.log(px.shift(-10)/px)
ics=[]; rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=pd.Series([q[1] for q in rows],index=[q[0] for q in rows]).dropna()
print('assets',len(C),'rows',len(px),'dates',len(a),'meanN',round(np.mean([q[2] for q in rows]),2),'coverage',round(len(a)/max(1,len(sig)-10),4))
print('IC %.8f ICIR %.8f hit %.6f turnover %.6f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
for label,lo,hi in [('2020-24','2020-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31'),('2028-30','2028-01-01','2030-05-29')]:
 q=a.loc[lo:hi]; print(label,len(q),round(q.mean(),8),round(q.mean()/q.std(ddof=1),8) if len(q)>1 else np.nan)
for h in [1,5,20]:
 yy=np.log(px.shift(-h)/px); aa=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(aa),8),len(aa))
sig.loc[a.index].to_csv('scripts/miner_3_20300530_stress_weighted_reversal_signal.csv')
