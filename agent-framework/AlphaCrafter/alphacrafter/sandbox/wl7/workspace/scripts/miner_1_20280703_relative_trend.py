import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d): break
  except: pass
 if d is not None:
  z=d[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Relative trend breadth: asset 20d return minus contemporaneous cross-sectional median,
# scaled by its 40d volatility and lagged one session.
raw=r.rolling(20,min_periods=15).sum(); med=raw.median(axis=1); vol=r.rolling(40,min_periods=25).std()
sig=((raw.sub(med,axis=0))/(vol*np.sqrt(20)+1e-8)).shift(1).rank(axis=1,pct=True)
for h in (1,5,10,20):
 f=p.shift(-h)/p-1; vals=[];dates=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x):vals.append(x);dates.append(dt);ns.append(len(z))
 a=np.array(vals);print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 if h==10:
  print('coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(U)),4),'turnover',round(sig.diff().abs().mean().mean(),6),'assets',len(D))
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31')]:
   q=a[[pd.Timestamp(lo)<=x<=pd.Timestamp(hi) for x in dates]];print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),4) if len(q)>1 else None)
sig.to_csv('scripts/miner_1_20280703_relative_trend_signal.csv',index_label='date');print('range',p.index.min(),p.index.max())
