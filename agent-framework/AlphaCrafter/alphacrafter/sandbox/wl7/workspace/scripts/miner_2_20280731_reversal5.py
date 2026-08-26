import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];fs={}
for s in u:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=3000)
  except:pass
  if d is not None and len(d):break
 if d is not None and len(d):
  x=d[['date','close']].dropna();x.date=pd.to_datetime(x.date);fs[s]=x.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(fs).sort_index().ffill(); r=p.pct_change(); sig=(-r.rolling(5).sum()).shift(1)
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1;v=[];ds=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):v.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(v);print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 if h==10:
  rk=sig.rank(axis=1,pct=True);t=[]
  for i in range(1,len(rk)):
   c=rk.iloc[i].dropna().index.intersection(rk.iloc[i-1].dropna().index)
   if len(c)>=8:t.append(abs(rk.iloc[i][c]-rk.iloc[i-1][c]).mean())
  print('TURN',round(np.mean(t),6),'COVERAGE',round(sig.notna().sum().sum()/(sig.shape[0]*len(u)),4))
  for lab,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31'),('2028','2028','2028-12-31')]:
   q=np.array([v[i] for i,d in enumerate(ds) if pd.Timestamp(a)<=d<=pd.Timestamp(b)]);print('REG',lab,'n',len(q),'IC',round(q.mean(),6))
sig.index.name='date';sig.to_csv('scripts/miner_2_20280731_reversal5_signal.csv');print('range',p.index.min(),p.index.max(),'assets',len(fs))
