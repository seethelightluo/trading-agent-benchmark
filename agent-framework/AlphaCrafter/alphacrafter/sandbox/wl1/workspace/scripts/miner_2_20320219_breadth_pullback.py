import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-02-05'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff()
# Breadth-confirmed pullback: reverse recent 5d relative return, but favor assets with
# persistent positive 20d daily breadth; scale by 30d realized risk. Shift one day.
rel=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0)
breadth=(r.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
f=(-rel/(vol+1e-9))*breadth
f=f.shift(1)
fr={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}
for h,fw in fr.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 s=pd.Series(vals); print('h',h,'dates',len(s),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4))
# daily rank turnover proxy
turn=[]
for i in range(1,len(f)):
 z=f.iloc[i-1:i+1].dropna(axis=1)
 if z.shape[1]>=8: turn.append((z.iloc[0].rank()-z.iloc[1].rank()).abs().mean()/z.shape[1])
print('turnover',round(np.mean(turn),6),'dates',len(px),'assets',len(px.columns))
# regimes for 10d
fw=fr[10]; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
s=pd.Series(dict(vals))
for a,b in [('2026-01-01','2028-12-31'),('2029-01-01','2030-12-31'),('2031-01-01','2032-02-05')]:
 q=s[(s.index>=a)&(s.index<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320219_breadth_pullback_signal.csv',index=False); print('artifact',len(out))
