import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(40).std()*np.sqrt(252)
# medium trend with a recent reversal penalty: 60d return minus 0.5*10d return, normalized by 40d risk
f=((px.pct_change(60)-0.5*px.pct_change(10))/vol); f=f.sub(f.mean(axis=1),axis=0)
y=px.shift(-10)/px-1;a=[];ns=[];ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
ic=pd.Series(a,index=ds).dropna()
print('candidate=medium_trend_recent_reversal_risk_10d');print('assets',len(P),'rows',len(px),'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-17')]:
 z=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
 if len(z):print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300418_medium_trend_signal.csv',index=False)
