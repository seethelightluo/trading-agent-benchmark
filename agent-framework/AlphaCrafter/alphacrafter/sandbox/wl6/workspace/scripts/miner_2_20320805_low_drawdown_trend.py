import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-08-04')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
r=P.pct_change()
# Trend quality: 60d return, penalized by realized volatility and by peak-to-trough drawdown.
ret=P/P.shift(60)-1
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252)
dd=(P/P.rolling(60,min_periods=40).max()-1).clip(upper=0).abs()
f=(ret/(vol+1e-12))*(1-dd.clip(0,1))
print('cutoff',CUT.date(),'rows',len(P),'assets',len(A),'coverage',round(f.notna().stack().mean(),6))
for h in [5,10,20]:
 ic=[]; ns=[]; dates=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): ic.append(q); ns.append(len(z)); dates.append(P.index[i])
 x=np.array(ic)
 print('horizon',h,'valid_dates',len(x),'avg_n',round(np.mean(ns),3),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4))
 print('regimes',{int(y):round(float(np.mean([v for v,d in zip(x,dates) if d.year==y])),6) for y in sorted(set(d.year for d in dates))})
q=f.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean(axis=1).dropna().mean(),6))
