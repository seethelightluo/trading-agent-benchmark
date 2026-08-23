import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-08-04')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
r=P.pct_change()
# Continuous breadth-confirmed slow trend: 90d risk-adjusted return
# multiplied by a smooth 60d fraction of instruments with positive returns.
ret90=P/P.shift(90)-1
vol90=r.rolling(90,min_periods=60).std()*np.sqrt(90)
breadth=((P/P.shift(20)-1)>0).rolling(60,min_periods=40).mean().mean(axis=1)
f=(ret90/vol90).mul(0.5+breadth,axis=0)
print('cutoff',CUT.date(),'dates',len(P),'assets',len(A),'raw_coverage',round(f.notna().stack().mean(),6))
for h in [5,10,20,40]:
 ic=[]; ns=[]; dates=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): ic.append(q);ns.append(len(z));dates.append(P.index[i])
 x=np.asarray(ic); print('horizon',h,'valid_dates',len(x),'avg_n',round(np.mean(ns),3),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4))
 if h==10: print('regimes',{int(y):round(float(np.mean([v for v,d in zip(x,dates) if d.year==y])),6) for y in sorted(set(d.year for d in dates))})
q=f.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean(axis=1).dropna().mean(),6))
