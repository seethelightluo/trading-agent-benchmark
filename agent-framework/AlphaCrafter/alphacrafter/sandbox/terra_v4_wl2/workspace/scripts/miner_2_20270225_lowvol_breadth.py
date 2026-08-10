import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-25'); F={}
for s in U:
 d=get_stock_daily_data(s,2200)
 if d is None:d=get_index_daily_data(s,2200)
 if d is not None:
  d=d[pd.to_datetime(d.date)<=cut].copy(); d.date=pd.to_datetime(d.date); F[s]=d.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(F).sort_index(); r=p.pct_change()
# Low-volatility factor, conditioned on positive breadth (avoid buying volatility in broad selloffs)
vol=r.rolling(20,min_periods=15).std(); breadth=(r.rolling(5).mean()>0).mean(axis=1)
f=-vol
f=f.where(breadth.shift(1)>=.5)
for h in [1,5,10]:
 a=[]; ns=[]
 for i,dt in enumerate(p.index):
  if i+h>=len(p):continue
  z=pd.concat([f.loc[dt],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(a).dropna(); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4),'cov',round(np.mean(ns)/15,4))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.date=out.date.dt.strftime('%Y-%m-%d');out.to_csv('../persistent/factor_signals_miner_2_20270225_lowvol_breadth.csv',index=False);print('artifact',len(out),out.date.nunique())
