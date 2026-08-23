import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); mom=px.pct_change(20); vol=r.rolling(20).std()*np.sqrt(252)
disp=r.sub(r.mean(axis=1),axis=0).abs().mean(axis=1); q=disp.rolling(252,min_periods=100).quantile(.40)
# In calm cross-asset regimes, favor persistent intermediate momentum, risk normalized.
f=(mom/vol).sub((mom/vol).mean(axis=1),axis=0).where(disp<q,np.nan)
y=px.shift(-5)/px-1; a=[];ns=[];ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
ic=pd.Series(a,index=ds).dropna()
print('candidate=low_dispersion_risk_momentum_20d'); print('assets',len(P),'rows',len(px),'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-17')]:
 z=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
 if len(z): print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('state_fraction',round((disp<q).mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300418_lowdisp_risk_momentum_signal.csv',index=False)
