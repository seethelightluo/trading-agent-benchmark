import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); prices[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(prices).sort_index().ffill(); r=np.log(p).diff()
ret60=np.log(p/p.shift(60)); posfrac=(r>0).rolling(60,min_periods=45).mean(); down=(-r.clip(upper=0)).rolling(40,min_periods=30).std(); f=(ret60*(.5+posfrac)/down.replace(0,np.nan)).shift(1)
for h in [1,5,10,20]:
 fr=np.log(p.shift(-h)/p); q=[]; ns=[]; ts=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ranks=f.loc[dt].rank(pct=True).reindex(U).fillna(.5)
   if prev is not None: ts.append((ranks-prev).abs().mean())
   prev=ranks
 q=pd.Series(q).dropna(); print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),round(np.mean(q>0),4),round(np.mean(ts),4))
fr=np.log(p.shift(-20)/p); q=[];ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt)
q=pd.Series(q,index=pd.to_datetime(ds));
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2030-12-31'),('2031','2031-12-31')]:
 z=q.loc[a:b]; print('regime',a,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6) if len(z)>1 else None)
f.to_csv('scripts/miner_3_20311030_trend_consistency_volscaled_signal.csv',index_label='date');print('range',p.index.min(),p.index.max(),'assets',len(p.columns),'coverage',round(f.notna().sum().mean()/len(U),4))
