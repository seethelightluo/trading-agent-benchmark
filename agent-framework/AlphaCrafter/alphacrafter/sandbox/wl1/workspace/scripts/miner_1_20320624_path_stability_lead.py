import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-06-23'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); resid=r-r.mean(axis=1).values[:,None]
rel20=resid.rolling(20,min_periods=15).sum(); rel60=resid.rolling(60,min_periods=40).sum(); down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=25).mean()); pos=(r>0).rolling(40,min_periods=25).mean()
f=((.55*rel60+.45*rel20)/(down+1e-8)*(.5+.5*pos)).shift(1); fr=np.log(px.shift(-10)/px); f=-f
vals=[]; ns=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals); print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avgN',np.mean(ns),'coverage',f.notna().mean().mean(),'turn',np.mean(turns)); print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 v=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v); print(a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan,'hit',np.mean(q>0) if len(q) else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320624_path_stability_lead_signal.csv',index=False)
