import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):break
  except Exception:pass
 if d is not None:
  x=d[['date','close','volume']].copy();x.date=pd.to_datetime(x.date).dt.normalize();D[s]=x.drop_duplicates('date').set_index('date').sort_index()
P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill(); V=pd.DataFrame({s:d.volume for s,d in D.items()}).sort_index().ffill(); r=P.pct_change();
# Recent return confirmed by abnormal volume, with volatility normalization.
vs=V/(V.rolling(40,min_periods=20).median()+1e-12); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
sig=(P.pct_change(5)*np.log1p(vs)/(vol+.01)).shift(1)
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; a=[];ds=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 if h==10:
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'coverage',round(sig.notna().sum().sum()/(len(sig)*len(U)),4),'assets',len(D))
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
   q=np.array([v for v,d in zip(a,ds) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]);print('REG',lab,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6))
sig.index.name='date';sig.reset_index().to_csv('scripts/miner_3_20270827_volume_confirmed_momentum_signal.csv',index=False)
