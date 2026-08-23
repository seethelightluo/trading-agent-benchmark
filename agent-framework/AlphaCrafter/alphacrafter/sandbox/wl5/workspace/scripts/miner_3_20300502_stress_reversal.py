import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
# High-stress conditional short-term reversal: reversal is stronger when VIX is elevated
v=pd.read_csv('../persistent/index_data/VIX.csv');v.columns=[str(x).lower() for x in v.columns]
dcol='date'; pcol='close' if 'close' in v.columns else [x for x in v.columns if x not in ('date','datetime')][0]
v[dcol]=pd.to_datetime(v[dcol]); vx=v.set_index(dcol)[pcol].astype(float).reindex(px.index).ffill()
# causal stress multiplier, clipped to avoid extreme concentration
stress=(vx/vx.rolling(120,min_periods=60).median()).clip(0.7,1.8)
f=(-px.pct_change(5)/vol).mul(stress,axis=0)
f=f.sub(f.mean(axis=1),axis=0)
y=px.shift(-5)/px-1
ics=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
ic=pd.Series(ics,index=dates).dropna()
print('candidate=stress_conditioned_short_reversal_5d')
print('assets',len(P),'rows',len(px),'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-01')]:
 z=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
 if len(z):print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
# decay diagnostics
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; aa=[];dd=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dd.append(dt)
 q=pd.Series(aa,index=dd).dropna();print('decay',h,round(q.mean(),6),len(q))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300502_stress_reversal_signal.csv',index=False)
