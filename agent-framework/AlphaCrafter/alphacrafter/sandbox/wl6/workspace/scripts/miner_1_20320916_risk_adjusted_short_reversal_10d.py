import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-09-15')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Contrarian recent move, normalized by instrument risk; smooth with 20d drawdown context
f=-(P/P.shift(5)-1)/vol
f=f*(1+0.25*(P/P.rolling(60,min_periods=30).max()-1))
print('cutoff',CUT.date(),'dates',len(P),'assets',len(A),'coverage',round(f.notna().stack().mean(),6))
for h in [5,10,20,40]:
 ic=[]; ns=[]; ds=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): ic.append(q); ns.append(len(z)); ds.append(P.index[i])
 x=np.array(ic); print('horizon',h,'valid_dates',len(x),'avg_n',round(np.mean(ns),3),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4))
 if h==10: print('regimes',{int(y):round(float(np.mean([v for v,d in zip(x,ds) if d.year==y])),6) for y in sorted(set(d.year for d in ds))})
q=f.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean(axis=1).dropna().mean(),6))
