import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=None
 try:d=get_stock_daily_data(s,5000)
 except:pass
 if d is None:
  try:d=get_index_daily_data(s,5000)
  except:pass
 if d is not None and len(d):
  x=d.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float).rename(s)
px=pd.concat(D,axis=1).sort_index();r=px.pct_change(); mom=px.pct_change(20)
# Macro-conditioned medium momentum: rank relative 20d return; follow in calm VIX,
# fade in stressed VIX. VIX is lagged through prior completed date.
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date')['close'].astype(float).reindex(px.index).ffill()
vr=v/v.rolling(120,min_periods=60).median(); sign=np.where(vr<1,1.,-1.)
sig=mom.sub(mom.median(axis=1),axis=0).mul(sign,axis=0)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15));print('daily',q.mean(),q.std(),q.mean()/q.std(),(q>0).mean())
for h in [3,5,10,20]:
 y=px.pct_change(h).shift(-h);vv=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=pd.Series(vv);print('h',h,s.mean(),s.mean()/s.std(),len(s))
for nm,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 s=a.loc[sl,'ic'].dropna();print(nm,len(s),s.mean(),s.mean()/s.std() if len(s)>1 else np.nan)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20301226_vix_conditioned_momentum_signal.csv',index=False)
