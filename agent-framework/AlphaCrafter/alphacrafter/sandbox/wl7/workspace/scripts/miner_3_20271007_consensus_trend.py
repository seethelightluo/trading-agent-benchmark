import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=3000)
 except Exception:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=3000)
  except Exception:pass
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date)
  F[s]=x.dropna().drop_duplicates('date').set_index('date').close.sort_index()
px=pd.DataFrame(F).sort_index().ffill(); r=px.pct_change()
# Consensus of medium/long trend, each normalized by its own realized volatility; lag one session.
def zrank(x): return x.rank(axis=1,pct=True)
parts=[]
for w in (20,60,120):
 vol=r.rolling(w,min_periods=max(10,w//2)).std()*np.sqrt(252)
 parts.append(zrank(px.pct_change(w)/(vol+0.01)))
sig=sum(parts)/len(parts); sig=sig.shift(1)
for h in (1,5,10,20):
 f=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q);dates.append(d);ns.append(len(z))
 a=np.array(vals)
 print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 if h==10:
  t=[]
  for i in range(1,len(sig)):
   c=sig.iloc[i].dropna().index.intersection(sig.iloc[i-1].dropna().index)
   if len(c)>=8:t.append(np.mean(abs(sig.iloc[i][c].rank(pct=True)-sig.iloc[i-1][c].rank(pct=True))))
  print('TURN',round(np.mean(t),6),'coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(u)),4),'assets',len(F))
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
   q=a[[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in dates]]
   print('REG',lab,'n',len(q),'IC',round(q.mean(),6))
sig.to_csv('scripts/miner_3_20271007_consensus_trend_signal.csv',index_label='date')
print('range',px.index.min(),px.index.max())
