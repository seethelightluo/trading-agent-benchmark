import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
R=P.pct_change(); stress=(v/v.rolling(60,min_periods=30).median()-1).shift(1).clip(0,2)
F=(-R.rolling(3,min_periods=3).sum().shift(1)).mul(1+stress,axis=0)
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; ics=[]; ns=[]; prev=None; turns=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fr.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8:
   ics.append(spearmanr(q.f,q.r).statistic); ns.append(len(q))
   rr=q.f.rank().reindex(assets)
   if prev is not None: turns.append(np.mean((rr-prev).abs().dropna()>0))
   prev=rr
 a=np.array(ics); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turn',np.nanmean(turns))
fr=P.shift(-10)/P-1; z=[]
for d in P.index:
 q=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
 if len(q)>=8:z.append((d,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
a=pd.DataFrame(z,columns=['date','ic'])
for lo,hi in [('2020','2026-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2035-12-31')]:
 q=a[(a.date>=lo)&(a.date<=hi)].ic; print('REG',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
F.to_csv('scripts/miner_2_20350216_stress_reversal_signal.csv')
