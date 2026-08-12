import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: d=get_stock_daily_data(s, days=5000)
 except: d=None
 if d is None or len(d)<300:
  try: d=get_index_daily_data(s, days=5000)
  except: d=None
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float).rename(s)
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
mom20=px/px.shift(20)-1; mom60=px/px.shift(60)-1; vol=r.rolling(20).std()*np.sqrt(252)
sig=(mom20-mom60)/vol.replace(0,np.nan); sig=sig.sub(sig.median(axis=1),axis=0); fwd=r.shift(-1)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),z.iloc[:,0].corr(z.iloc[:,1])))
a=pd.DataFrame(rows,columns=['date','n','ic_rank','ic_pearson']).set_index('date')
for col in ['ic_rank','ic_pearson']:
 print(col,'mean',a[col].mean(),'std',a[col].std(),'ICIR',a[col].mean()/a[col].std(),'hit',(a[col]>0).mean())
print('dates',len(a),'median n',a.n.median(),'coverage',a['n'].sum()/(len(a)*15))
for name,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 q=a.loc[sl,'ic_rank']; print(name,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,3,5,10,20]:
 y=px.pct_change(h).shift(-h); vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals); print('horizon',h,'IC',q.mean(),'ICIR',q.mean()/q.std(),'n',len(q))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20301212_trend_acceleration_signal.csv',index=False)
