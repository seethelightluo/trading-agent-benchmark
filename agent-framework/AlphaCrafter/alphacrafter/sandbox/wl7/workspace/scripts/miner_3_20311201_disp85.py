import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U};P=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill();R=np.log(P).diff();ret=P.pct_change(20);disp=ret.std(axis=1);thr=disp.rolling(120,min_periods=60).quantile(.85)
base=-(ret.sub(ret.median(axis=1),axis=0))/(R.rolling(40,min_periods=30).std()*np.sqrt(252));sig=base.where(disp.gt(thr),np.nan).shift(1)
for h in [1,5,10]:
 y=P.pct_change(h).shift(-h);a=[];ns=[];ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):a.append(c);ns.append(len(z));ds.append(dt)
 a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round(np.mean(a>0),5),'datecov',round(len(a)/len(P),5))
 if h==5:pd.DataFrame({'date':ds,'ic':a,'n':ns}).to_csv('scripts/miner_3_20311201_disp85_ic.csv',index=False)
print('history_dates',len(P),'assets',len(P.columns),'coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20311201_disp85_signal.csv',index=False)
