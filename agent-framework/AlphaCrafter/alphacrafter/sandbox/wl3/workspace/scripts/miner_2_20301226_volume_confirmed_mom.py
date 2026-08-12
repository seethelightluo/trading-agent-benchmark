import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in U:
 d=None
 try:d=get_stock_daily_data(s,5000)
 except:pass
 if d is None:
  try:d=get_index_daily_data(s,5000)
  except:pass
 if d is not None and len(d):
  x=d.copy();x.date=pd.to_datetime(x.date); z=x.set_index('date')
  P[s]=z.close.astype(float).rename(s)
  if 'volume' in z: V[s]=z.volume.astype(float).replace(0,np.nan).rename(s)
px=pd.concat(P,axis=1).sort_index(); vol=pd.concat(V,axis=1).reindex(px.index)
r=px.pct_change(); r10=px.pct_change(10)
# Volume-confirmed medium momentum: relative 10-session return, strengthened by
# log short/long volume surprise. Cross-sectional demean avoids market direction.
vs=np.log(vol.rolling(10,min_periods=5).mean()/vol.rolling(60,min_periods=30).mean())
vs=vs.clip(-3,3)
base=r10.sub(r10.median(axis=1),axis=0)
sig=base*(1+0.5*vs)
sig=sig.replace([np.inf,-np.inf],np.nan)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
  if pd.notna(c):rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15))
print('daily IC',q.mean(),'std',q.std(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
for h in [3,5,10,20]:
 y=px.pct_change(h).shift(-h);vv=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):vv.append(c)
 s=pd.Series(vv);print('h',h,'IC',s.mean(),'ICIR',s.mean()/s.std(),'n',len(s))
for nm,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 s=a.loc[sl,'ic'].dropna();print(nm,len(s),s.mean(),s.mean()/s.std() if len(s)>1 else np.nan)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20301226_volume_confirmed_mom_signal.csv',index=False)
