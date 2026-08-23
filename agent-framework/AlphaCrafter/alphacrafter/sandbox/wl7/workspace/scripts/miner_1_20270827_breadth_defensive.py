import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except Exception: pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except Exception: pass
 if d is not None and len(d):
  q=d[['date','close']].copy();q.date=pd.to_datetime(q.date);F[s]=q.dropna().drop_duplicates('date').set_index('date').close.sort_index()
px=pd.DataFrame(F).sort_index().ffill(); ret=px.pct_change(); r20=px.pct_change(20)
vol=ret.rolling(30,min_periods=25).std()*np.sqrt(252)
# Defensive-relative-strength regime factor: lagged trend, with breadth gate and defensive sleeve tilt.
trend=(r20/(vol+.01)).shift(1)
breadth=(r20>0).mean(axis=1).rolling(10,min_periods=5).mean().shift(1)
defs=['XAU','US10Y','CN10Y']; dscore=trend[defs].mean(axis=1)
# In weak breadth, favor defensive leaders; in broad strength, retain diversified trend.
gate=(breadth<0.45).astype(float)
sig=trend.copy()
sig=sig.mul(1-gate,axis=0).add(trend.mul(gate,axis=0).mul(0.35),fill_value=0)
sig.loc[gate>0,defs]=sig.loc[gate>0,defs].add(dscore[gate>0].values[:,None]*0.65)
sig.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20270827_breadth_defensive_signal.csv',index=False)
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c);dates.append(dt);ns.append(len(z))
 a=np.array(vals); ir=a.mean()/a.std(ddof=1)*np.sqrt(len(a))
 print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
 if h==10:
  turns=[]
  for i in range(1,len(sig)):
   z=sig.iloc[i].dropna().index.intersection(sig.iloc[i-1].dropna().index)
   if len(z)>=8: turns.append(np.mean(abs(sig.iloc[i][z].rank(pct=True)-sig.iloc[i-1][z].rank(pct=True))))
  print('TURN',round(np.mean(turns),6),'coverage',round(sig.notna().sum().sum()/(sig.shape[0]*15),4),'assets',len(F))
  for lab,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31')]:
   q=a[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]
   print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6) if len(q)>1 and q.std(ddof=1)>0 else None)
print('range',px.index.min(),px.index.max())
