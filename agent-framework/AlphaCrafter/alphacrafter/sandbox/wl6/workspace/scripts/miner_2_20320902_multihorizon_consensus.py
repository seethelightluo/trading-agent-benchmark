import pandas as pd,numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-09-01')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A for _ in [0]}).sort_index().loc[:CUT]
# consensus trend: equal-weight medium/long returns, penalize disagreement and recent reversal.
r20=P/P.shift(20)-1; r60=P/P.shift(60)-1; r120=P/P.shift(120)-1
# rank transforms prevent price-scale effects; agreement is fraction of horizons with same sign
z=pd.concat([r20.rank(axis=1,pct=True),r60.rank(axis=1,pct=True),r120.rank(axis=1,pct=True)]).groupby(level=0).mean()
signs=(pd.concat([r20>0,r60>0,r120>0]).groupby(level=0).mean())
agree=1-abs(signs-0.5)*2
f=z*(0.5+0.5*agree)
print('cutoff',CUT.date(),'dates',len(P),'assets',len(A),'raw_coverage',round(f.notna().stack().mean(),6))
for h in [5,10,20,40]:
 ic=[];ns=[];ds=[]
 for i in range(len(P)-h):
  q=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.x,q.y).statistic
   if np.isfinite(v):ic.append(v);ns.append(len(q));ds.append(P.index[i])
 x=np.array(ic); print('horizon',h,'valid_dates',len(x),'avg_n',round(np.mean(ns),3),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4))
 if h==10: print('regimes',{int(y):round(float(np.mean([v for v,d in zip(x,ds) if d.year==y])),6) for y in sorted(set(d.year for d in ds))})
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
