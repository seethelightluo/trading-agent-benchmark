import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except:pass
 if d is not None and len(d):
  x=d[['date','close']].dropna().drop_duplicates('date');x.date=pd.to_datetime(x.date);frames[s]=x.sort_values('date').set_index('date').close
px=pd.DataFrame(frames).sort_index().ffill(); r=px.pct_change()
# Market-neutral trend: 30d asset return net of rolling 60d beta times equal-weight cross-asset return, scaled by residual downside risk.
mkt=r.mean(axis=1); cov=r.rolling(60,min_periods=45).cov(mkt); var=mkt.rolling(60,min_periods=45).var(); beta=cov.div(var,axis=0)
res=r.sub(beta.mul(mkt,axis=0)); down=res.where(res<0).rolling(30,min_periods=20).std()*np.sqrt(252)
sig=(res.rolling(30,min_periods=20).sum()/(down+0.02)).shift(1).rank(axis=1,pct=True)
print('range',px.index.min(),px.index.max(),'assets_loaded',len(frames),'universe',len(U))
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):vals.append(q);ds.append(dt);ns.append(len(z))
 v=np.asarray(vals); ir=v.mean()/v.std(ddof=1)*np.sqrt(len(v))
 print('H',h,'dates',len(v),'avg_n',round(np.mean(ns),2),'IC',round(v.mean(),6),'ICIR',round(ir,6),'hit',round(np.mean(v>0),4))
 if h==10:
  tr=[]
  for i in range(1,len(sig)):
   a=sig.iloc[i].dropna();b=sig.iloc[i-1].dropna(); c=a.index.intersection(b.index)
   if len(c)>=8:tr.append(np.mean(abs(a[c]-b[c])))
  print('TURN',round(np.mean(tr),6),'coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(U)),4))
  for lab,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
   q=np.asarray([vals[i] for i,d in enumerate(ds) if pd.Timestamp(a)<=d<=pd.Timestamp(b)]); z=q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else 0
   print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(z,6))
# save artifact for audit
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270823_residual_beta_trend_signal.csv',index=False)
