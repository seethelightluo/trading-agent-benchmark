import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in watch:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x['date']=pd.to_datetime(x['date']); prices[s]=x.drop_duplicates('date').set_index('date')['close'].astype(float)
px=pd.DataFrame(prices).sort_index().ffill(); r=np.log(px).diff(); mom=np.log(px/px.shift(60)); vol20=r.rolling(20).std()*np.sqrt(252); vol60=r.rolling(60).std()*np.sqrt(252)
raw=mom/(vol60+1e-8); cs=raw.sub(raw.mean(axis=1),axis=0); compression=(vol20/(vol60+1e-8)).clip(.5,2); signal=(cs*(1+.75*(1-compression))).shift(1)
for h in [5,10,20,40,60]:
 fwd=np.log(px.shift(-h)/px); vals=[]; ns=[]; dates=[]
 for dt in signal.index:
  a=pd.concat([signal.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q); ns.append(len(a)); dates.append(dt)
 z=pd.Series(vals,index=dates); print('horizon',h,'dates',len(z),'avg_n %.2f min_n %d IC %.8f ICIR %.8f hit %.4f'%(np.mean(ns),min(ns),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 if h==20:
  for name,start,end in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-10-01')]:
   u=z.loc[start:end]; print(name,len(u),'IC %.8f ICIR %.8f'%(u.mean(),u.mean()/u.std(ddof=1)) if len(u)>1 else 'NA')
out=signal.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_2_20311002_volatility_conditioned_trend_signal.csv',index=False)
print('assets',len(prices),'coverage %.6f'%(signal.notna().mean().mean()))
