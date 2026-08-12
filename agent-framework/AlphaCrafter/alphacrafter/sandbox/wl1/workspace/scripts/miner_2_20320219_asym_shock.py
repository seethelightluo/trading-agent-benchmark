import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-02-05');raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);raw[s]=d[d.date<=cut].sort_values('date').set_index('date').close
px=pd.DataFrame(raw).sort_index();r=np.log(px).diff(); v=r.rolling(20,min_periods=15).std()
# Asymmetric shock-rebound: reverse unusually negative 3d return, penalize downside risk,
# and require a positive 60d trend to avoid catching persistent collapse.
z3=r.rolling(3).sum(); cs=z3.sub(z3.median(axis=1),axis=0); shock=np.minimum(cs,0)
trend=np.tanh(np.log(px/px.shift(60))*2); downside=np.sqrt((r.clip(upper=0)**2).rolling(30,min_periods=20).mean())
f=(-shock/(downside*np.sqrt(3)+1e-9))*((trend+1)/2); f=f.shift(1)
for h in [1,5,10,20]:
 fw=np.log(px.shift(-h)/px);vals=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 s=pd.Series(vals); print('h',h,'dates',len(s),'avg_n',round(np.mean(ns),3),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4))
turn=[]
for i in range(1,len(f)):
 q=f.iloc[i-1:i+1].dropna(axis=1)
 if q.shape[1]>=8:turn.append((q.iloc[0].rank()-q.iloc[1].rank()).abs().mean()/q.shape[1])
print('turnover',round(np.mean(turn),6),'assets',len(px.columns),'dates',len(px))
fw=np.log(px.shift(-10)/px);vals=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(q)>=8:vals.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
s=pd.Series(dict(vals))
for a,b in [('2026-01-01','2028-12-31'),('2029-01-01','2030-12-31'),('2031-01-01','2032-02-05')]:
 q=s[(s.index>=a)&(s.index<=b)];print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320219_asym_shock_signal.csv',index=False);print('artifact',len(out))
