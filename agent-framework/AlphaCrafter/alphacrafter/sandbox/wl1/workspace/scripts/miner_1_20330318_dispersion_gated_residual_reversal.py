import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-03-17'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close.astype(float)
P=pd.DataFrame(raw).sort_index(); lr=np.log(P).diff(); res=lr.sub(lr.mean(axis=1),axis=0)
# Reversal of a medium residual move, activated only when cross-sectional dispersion is elevated.
disp=res.std(axis=1).rolling(20,min_periods=15).rank(pct=True)
base=-(res.rolling(20,min_periods=15).sum()/res.rolling(40,min_periods=25).std())
f=base.mul((disp>0.60).astype(float),axis=0).shift(1)
fr=np.log(P.shift(-10)/P); ics=[]; ns=[]; turns=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): ics.append(c); ns.append(len(z))
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank(pct=True).sub(q.iloc[:,1].rank(pct=True)).abs().mean())
a=np.array(ics); print('candidate dispersion_gated_residual_reversal_20d'); print('assets',len(raw),'calendar_dates',len(P),'valid_dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(turns))
for a0,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=[]
 for i,d in enumerate(f.index):
  if a0<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c): q.append(c)
 q=np.array(q); print(a0+'-'+b,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20330318_dispersion_gated_residual_reversal_signal.csv',index=False)
