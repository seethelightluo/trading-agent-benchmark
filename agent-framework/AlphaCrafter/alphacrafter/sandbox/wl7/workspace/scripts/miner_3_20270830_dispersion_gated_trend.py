import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
acct=get_account_dict(); u=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except:pass
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);F[s]=x.dropna().drop_duplicates('date').set_index('date').close.sort_index()
px=pd.DataFrame(F).sort_index().ffill(); r=px.pct_change(); r20=px.pct_change(20); vol=r.rolling(30,min_periods=25).std()*np.sqrt(252)
# high cross-sectional dispersion selects trend; low dispersion selects short reversal
csdisp=r20.std(axis=1); med=csdisp.rolling(120,min_periods=60).median(); high=(csdisp>med).astype(float)
trend=(r20/(vol+0.01)).shift(1); rev=(-px.pct_change(5)/(r.rolling(20,min_periods=15).std()*np.sqrt(252)+.01)).shift(1)
sig=trend.mul(high,axis=0)+rev.mul(1-high,axis=0)
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):vals.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(vals); ir=a.mean()/a.std(ddof=1)*np.sqrt(len(a))
 print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
 if h==10:
  turn=[]
  for i in range(1,len(sig)):
   c=sig.iloc[i].dropna().index.intersection(sig.iloc[i-1].dropna().index)
   if len(c)>=8:turn.append(np.mean(abs(sig.iloc[i][c].rank(pct=True)-sig.iloc[i-1][c].rank(pct=True))))
  print('TURN',round(np.mean(turn),6),'coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(u)),4),'assets',len(F))
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
   q=a[[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in ds]]; ir2=q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 and q.std(ddof=1)>0 else 0
   print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(ir2,6))
print('range',px.index.min(),px.index.max())
