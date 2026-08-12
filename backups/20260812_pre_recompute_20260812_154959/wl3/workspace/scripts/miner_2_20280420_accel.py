import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<100:d=get_index_daily_data(s,days=4000)
 return d
p={}
for s in U:
 d=get(s)
 if d is not None:p[s]=d.set_index('date').close
px=pd.DataFrame(p).sort_index(); r=px.pct_change()
# acceleration: recent 20d return relative to prior 40d return, scaled by 60d volatility; lag one day
f=((px/px.shift(20)-1)-(px.shift(20)/px.shift(60)-1))/r.rolling(60,min_periods=40).std(); f=f.shift(1)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(a).dropna();print('H',h,'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),4),'hit',round((a>0).mean(),4),'dates',len(a),'avgN',round(np.mean(ns),2))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20280420_accel_signal.csv',index=False)
