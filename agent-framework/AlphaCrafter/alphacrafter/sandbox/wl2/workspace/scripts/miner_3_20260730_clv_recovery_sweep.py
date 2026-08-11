import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=D['SPX'].index; P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U},index=dates); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U},index=dates); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U},index=dates)
def rank(x): return x.rank(axis=1,pct=True)
clv=(-(2*(P-L)/(H-L))).shift(1); rec=(P/P.rolling(60,min_periods=40).min()-1).shift(1)
print('idea clv-recovery rank blend sweep; universe',len(U),'dates',len(dates))
for w in [0.05,0.10,0.15,0.25,0.40,0.60,1.0]:
 F=rank(clv)+w*rank(rec); q=[];ns=[]
 Y=P.shift(-1).div(P)-1
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   r=spearmanr(z.f,z.y).statistic
   if np.isfinite(r): q.append(r);ns.append(len(z))
 q=np.array(q); print('w',w,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
