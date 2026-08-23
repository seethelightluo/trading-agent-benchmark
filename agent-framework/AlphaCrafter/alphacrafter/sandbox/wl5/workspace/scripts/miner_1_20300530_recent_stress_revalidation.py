import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change();vol=r.rolling(20).std()*np.sqrt(252)
v=pd.read_csv('../persistent/index_data/VIX.csv');v.columns=[str(x).lower() for x in v.columns];v.date=pd.to_datetime(v.date); c='close' if 'close' in v else [x for x in v if x!='date'][0];vx=v.set_index('date')[c].astype(float).reindex(px.index).ffill();stress=(vx/vx.rolling(120,min_periods=60).median()).clip(.7,1.8)
raw=-px.pct_change(5);rel=raw.sub(raw.median(axis=1),axis=0);f=(rel/vol).mul(stress,axis=0);f=f.sub(f.mean(axis=1),axis=0); y=px.shift(-5)/px-1
A=[];D=[];N=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:A.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));D.append(dt);N.append(len(z))
ic=pd.Series(A,index=D).dropna();ic=ic[ic.index>=pd.Timestamp('2028-01-01')]
print('recent_stress_relative','assets',len(P),'dates',len(ic),'mean_n',round(np.mean(N[-len(ic):]),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px.loc['2028':]),4))
