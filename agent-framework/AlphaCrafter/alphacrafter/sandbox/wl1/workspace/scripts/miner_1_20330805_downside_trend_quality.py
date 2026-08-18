import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); e=r.sub(bench,axis=0)
# Trend quality: medium residual return, penalize downside volatility, require long-window directional confirmation.
down=e.clip(upper=0).pow(2).rolling(40,min_periods=25).mean().pow(.5)
short=e.rolling(20,min_periods=15).sum(); long=e.rolling(60,min_periods=40).sum()
f=((short/(down+1e-8)) * np.sign(long)).shift(1); fr=np.log(px.shift(-10)/px)
ics=[]; ns=[]; dates=[]; tr=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(d)
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: tr.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(ics,index=pd.to_datetime(dates)).dropna(); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'turn',round(np.mean(tr),4))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))]; print(a,b,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
for h in [5,10,20]:
 ff=np.log(px.shift(-h)/px); vv=[]
 for d in f.index:
  z=pd.concat([f.loc[d],ff.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'IC',round(np.nanmean(vv),6),'n',len(vv))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330805_downside_trend_quality_signal.csv',index=False)
