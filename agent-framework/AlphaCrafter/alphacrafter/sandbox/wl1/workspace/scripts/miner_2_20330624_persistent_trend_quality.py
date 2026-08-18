import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-06-23'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date);raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); res=r-r.mean(axis=1).values[:,None]
res=pd.DataFrame(res,index=r.index,columns=r.columns); v=res.rolling(40,min_periods=25).std()
# Persistent trend quality: medium residual momentum rewarded only when short and medium signs agree; risk adjusted.
m20=res.rolling(20,min_periods=15).sum(); m60=res.rolling(60,min_periods=40).sum()
f=(m20/(v*np.sqrt(20)+1e-8))*np.sign(m20*m60); f=f.shift(1); fr=np.log(px.shift(-10)/px)
vals=[];ns=[];turn=[];dates=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(d)
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8:turn.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals,index=pd.to_datetime(dates)).dropna();print('assets',len(raw),'dates',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4),'turn',round(np.mean(turn),4))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))];print(a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20330624_persistent_trend_quality_signal.csv',index=False)
