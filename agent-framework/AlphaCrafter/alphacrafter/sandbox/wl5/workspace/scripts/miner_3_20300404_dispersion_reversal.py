import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); rr=px.pct_change(); r5=px.pct_change(5); v=rr.rolling(20).std()*np.sqrt(252)
# high dispersion state: cross-sectional absolute deviation of daily returns, trailing percentile
csdisp=rr.sub(rr.mean(axis=1),axis=0).abs().mean(axis=1)
q=csdisp.rolling(252,min_periods=100).quantile(.60)
state=(csdisp>q)
f=(-r5/v).sub((-r5/v).mean(axis=1),axis=0).where(state, np.nan)
for h in [5,10,20]:
 y=px.shift(-h)/px-1; a=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 ic=pd.Series(a,index=ds).dropna();print('H',h,'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-03')]:
  z=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
  if len(z):print(' ',lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('state',state.mean(),'assets',len(P),'rows',len(px))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300404_dispersion_reversal_signal.csv',index=False)
