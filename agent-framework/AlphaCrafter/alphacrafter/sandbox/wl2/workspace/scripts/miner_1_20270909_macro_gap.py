import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index(); vc=v['close'].rolling(252,min_periods=60).quantile(.7).shift(1)
all=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); prev=x.close.shift(1); gap=(x.open-prev)/prev; atr=(x.high-x.low).rolling(20,min_periods=10).mean()/prev
 # high VIX amplifies reversal, low VIX leaves baseline
 f=-gap/atr; regime=(vc.reindex(x.index).ffill()>vc.reindex(x.index).ffill().median()).astype(float)
 for name,fac in [('high_only',f*regime),('amplified',f*(1+regime)),('low_only',f*(1-regime))]:
  all.append(pd.DataFrame({'date':x.index,'sym':s,'variant':name,'factor':fac,'fwd':x.close.shift(-1)/x.close-1}))
z=pd.concat(all).reset_index(drop=True).replace([np.inf,-np.inf],np.nan).dropna()
for name,g0 in z.groupby('variant'):
 a=[]; ns=[]
 for dt,g in g0.groupby('date'):
  if len(g)>=8:
   c=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(c): a.append(c);ns.append(len(g))
 a=np.array(a); print(name,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for lo,hi in [('2020','2023'),('2024','2026'),('2027-01','2027-08')]:
  q=g0[(g0.date>=lo)&(g0.date<=hi)]; b=[]
  for dt,g in q.groupby('date'):
   if len(g)>=8:
    c=spearmanr(g.factor,g.fwd).statistic
    if np.isfinite(c):b.append(c)
  print(lo,round(np.mean(b),5) if b else None,len(b))
