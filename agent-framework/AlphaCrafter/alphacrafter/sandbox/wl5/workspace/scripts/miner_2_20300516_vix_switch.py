import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(30).std()*np.sqrt(252)
# Observation-only VIX is lagged/current completed-day information, aligned causally.
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vix=vix.set_index('date').close.astype(float).reindex(px.index).ffill()
z=(vix-vix.rolling(252).mean())/vix.rolling(252).std()
# Stress intensity: high VIX or a sharp VIX rise shifts from trend to short-term reversal.
stress=((z.clip(-2,2)+2)/4).fillna(.5)
trend=px.pct_change(20)/vol
rev=-px.pct_change(5)/vol
f=trend.mul(1-stress,axis=0)+rev.mul(stress,axis=0)
f=f.sub(f.mean(axis=1),axis=0)
ics=[]; ns=[]; ds=[]
for dt in f.index:
 y=px.shift(-10)/px-1; q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(q)>=8: ics.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q)); ds.append(dt)
ic=pd.Series(ics,index=ds).dropna()
print('candidate=vix_switch_trend_reversal_10d'); print('assets',len(P),'rows',len(px),'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-16')]:
 q=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
 if len(q): print('regime',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
for h in [5,10,20]:
 y=px.shift(-h)/px-1; aa=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8: aa.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 aa=pd.Series(aa).dropna(); print('decay',h,round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300516_vix_switch_signal.csv',index=False)
