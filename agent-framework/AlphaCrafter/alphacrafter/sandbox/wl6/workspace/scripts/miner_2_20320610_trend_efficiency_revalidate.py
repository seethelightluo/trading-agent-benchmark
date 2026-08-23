import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index().loc[:'2032-06-09']
r20=P/P.shift(20)-1; path=P.pct_change().abs().rolling(20,min_periods=18).sum(); anchor=P/P.shift(60)-1
F=(r20/(0.01+path))*(1+anchor.clip(-.5,.5))
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1],'coverage',round(float(F.notna().stack().mean()),4))
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; ics=[]; ns=[]; turns=[]; prev=None; dates=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('x'),fr.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.x,z.y).statistic);ns.append(len(z));dates.append(dt)
  rr=F.loc[dt].rank(pct=True)
  if prev is not None:
   q=pd.concat([rr,prev],axis=1).dropna()
   if len(q): turns.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=rr
 a=np.array(ics); metrics={'horizon':h,'valid_dates':len(a),'avg_instruments':round(np.mean(ns),3),'IC':round(float(np.mean(a)),6),'ICIR':round(float(np.mean(a)/np.std(a,ddof=1)),6),'hit_ratio':round(float(np.mean(a>0)),4),'turnover':round(float(np.mean(turns)),6)}
 print(metrics)
 if h==20:
  q=pd.DataFrame({'ic':a},index=dates); print('regimes',q.groupby(q.index.year).ic.agg(['mean','count']).round(6).to_dict('index'))
