import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); raw=-(p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0)); vd=get_index_daily_data('VIX',4000); v=vd.set_index(pd.to_datetime(vd.date)).close.sort_index().reindex(p.index).ffill(); gate=v>v.shift(1).rolling(60,min_periods=30).median(); sig=raw.where(gate,0.0)
print('data',p.index.min().date(),p.index.max().date(),'assets',len(p.columns),'gate_frac',round(gate.mean(),4))
for h in [5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ns.append(len(q)); dates.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(dates)); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,st in [('online','2026-07-16'),('recent252','2028-09-20'),('2029','2029-01-01')]:
  z=a[a.index>=st]; print(lab,len(z),round(z.mean(),7),round(z.mean()/z.std(ddof=1),6),round(np.mean(z>0),4))
 if h in [5,10,20]: sig.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}).to_csv(f'scripts/miner_3_20291004_vix_gated_reversal_{h}d_signal.csv',index=False)
