import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=None
 for f in (get_stock_daily_data,get_index_daily_data):
  try:d=f(s,5000)
  except:pass
  if d is not None and len(d):break
 if d is not None and len(d):
  z=d.copy();z.date=pd.to_datetime(z.date);P[s]=z.set_index('date').close.astype(float)
px=pd.concat(P,axis=1).sort_index();r=px.pct_change()
def load(n):
 z=pd.read_csv('../persistent/index_data/'+n+'.csv');z.date=pd.to_datetime(z.date); c='close' if 'close' in z else 'Close';return z.set_index('date')[c].astype(float).reindex(px.index).ffill()
dxy=load('DXY');vix=load('VIX'); m=dxy.pct_change(); res=pd.DataFrame(index=px.index,columns=px.columns)
for s in px:
 cov=r[s].rolling(60,min_periods=30).cov(m);res[s]=r[s]-cov/m.rolling(60,min_periods=30).var().replace(0,np.nan)*m
sig=-res.rolling(5,min_periods=5).sum(); active=vix>vix.rolling(60,min_periods=30).median();sig=sig.where(active).sub(sig.where(active).median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15));print('daily',q.mean(),q.std(),q.mean()/q.std(),(q>0).mean())
for h in [5,10]:
 y=px.pct_change(h).shift(-h);x=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=pd.Series(x).dropna();print(h,x.mean(),x.mean()/x.std(),len(x))
for nm,sl in [('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 x=a.loc[sl,'ic'].dropna();print(nm,len(x),x.mean(),x.mean()/x.std())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20310123_stress_residual_signal.csv',index=False)