import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-11-26'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d['date']=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); mom=np.log(px/px.shift(60)); dv=r.rolling(40).std(); sv=r.rolling(20).std();
# continuous recovery-pullback: favor large medium trend relative to recent component, with moderate risk penalty
for name,f in {'pullback':-(mom-.7*np.log(px/px.shift(10)))/(dv+1e-9),'pullback2':-(mom-.5*np.log(px/px.shift(10)))/(dv+1e-9),'reversal':-np.log(px/px.shift(20))/(dv+1e-9)}.items():
 f=f.shift(1); fw=np.log(px.shift(-20)/px); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 q=pd.Series(dict(vals)).dropna(); print(name,'dates',len(q),'avg_n',np.mean(ns),'cov',np.mean(ns)/15,'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',np.mean(q>0))
 for h in [1,5,10,20]:
  v=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],np.log(px.shift(-h)/px).loc[dt]],axis=1).dropna()
   if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  print(h,np.nanmean(v))
 if name=='pullback2':
  out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20311127_pullback2_signal.csv',index=False)