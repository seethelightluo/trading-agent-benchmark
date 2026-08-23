import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-08-18')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
r=P.pct_change()
# Recovery quality: rebound from 60d trough, discounted when rebound is volatile/downside-heavy.
low60=P.rolling(60,min_periods=40).min()
recovery=P/low60-1
ret20=P/P.shift(20)-1
down60=np.sqrt((r.clip(upper=0)**2).rolling(60,min_periods=40).mean())*np.sqrt(20)
# rank-preserving, interpretable recovery adjusted by downside risk and recent direction
f=(recovery/(down60+1e-12))*(0.5+0.5*(ret20>0))
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
