import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except:pass
 if d is not None and len(d):
  x=d[['date','close']].dropna().drop_duplicates('date');x.date=pd.to_datetime(x.date);F[s]=x.sort_values('date').set_index('date').close
px=pd.DataFrame(F).sort_index().ffill(); r=px.pct_change()
# Volume-confirmed medium-term trend: prior 20d return, multiplied by relative volume participation, lagged one day.
vols={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except:pass
 if d is not None and len(d):
  x=d[['date','volume']].dropna().drop_duplicates('date');x.date=pd.to_datetime(x.date);vols[s]=x.sort_values('date').set_index('date').volume
v=pd.DataFrame(vols).reindex(px.index).ffill(); rel=v/(v.rolling(60,min_periods=30).median()+1e-12)
sig=(px.pct_change(20)*np.sqrt(rel.clip(0.5,2.0))).shift(1).rank(axis=1,pct=True)
print('range',px.index.min(),px.index.max(),'assets',len(F),'volume_assets',len(vols))
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[];ds=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):vals.append(q);ds.append(dt);ns.append(len(z))
 q=np.asarray(vals); ir=q.mean()/q.std(ddof=1)*np.sqrt(len(q))
 print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(ir,6),'hit',round(np.mean(q>0),4))
 if h==10:
  tr=[]
  for i in range(1,len(sig)):
   a=sig.iloc[i].dropna();b=sig.iloc[i-1].dropna();c=a.index.intersection(b.index)
   if len(c)>=8:tr.append(np.mean(abs(a[c]-b[c])))
  print('TURN',round(np.mean(tr),6),'coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(U)),4))
  for lab,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
   z=np.asarray([q[i] for i,d in enumerate(ds) if pd.Timestamp(a)<=d<=pd.Timestamp(b)]); ir2=z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else 0
   print('REG',lab,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir2,6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270823_volume_trend_signal.csv',index=False)
