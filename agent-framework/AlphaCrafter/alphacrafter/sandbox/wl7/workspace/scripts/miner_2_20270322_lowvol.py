import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Defensive low-volatility quality: inverse short volatility, conditioned on stable long volatility.
v20=r.rolling(20).std(); v60=r.rolling(60).std()
f=-(v20/(v60+1e-12))
# use factor available at t, forward return t to t+h
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]; ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank())); ns.append(len(z)); ds.append(dt)
 q=pd.Series(vals,index=pd.to_datetime(ds)).dropna()
 print('H',h,'N',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4),'hit',round((q>0).mean(),4))
 if h==1:q1=q
print('coverage',round(f.notna().sum().sum()/p.notna().sum().sum(),4),'assets',len(px),'dates',len(p))
ranks=f.rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean(axis=1).dropna().mean(),5))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31'),('2026-27','2026','2027-12-31')]:
 q=q1[(q1.index>=lo)&(q1.index<=hi)]; print(label,len(q),round(q.mean(),6) if len(q) else None,round(q.mean()/q.std(ddof=1)*np.sqrt(252),4) if len(q)>1 else None)
f.to_csv('scripts/miner_2_20270322_lowvol_signal.csv',index_label='date')
