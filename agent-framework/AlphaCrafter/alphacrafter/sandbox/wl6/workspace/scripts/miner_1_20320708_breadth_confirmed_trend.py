import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-07-07')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
r=P.pct_change()
# Breadth-confirmed trend: blend medium and long momentum, then scale by market breadth.
m10=P/P.shift(10)-1; m40=P/P.shift(40)-1
breadth=(r.rolling(20,min_periods=15).apply(lambda x: np.mean(x>0),raw=True))
# use contemporaneous completed-date breadth, centered so neutral breadth does not alter rank strongly
breadth_signal=(breadth.mean(axis=1)-0.5)
f=(0.65*m10+0.35*m40).mul((1+0.8*breadth_signal).clip(0.55,1.45),axis=0)
print('cutoff',CUT.date(),'data_dates',len(P),'assets',len(A),'raw_coverage',round(f.notna().stack().mean(),6))
for h in [5,10,20]:
 ic=[]; ns=[]; dates=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): ic.append(q); ns.append(len(z)); dates.append(P.index[i])
 x=np.asarray(ic); print('horizon',h,'valid_dates',len(x),'avg_n',round(np.mean(ns),3),'IC',round(float(np.mean(x)),6),'ICIR',round(float(np.mean(x)/np.std(x,ddof=1)),6),'hit',round(float(np.mean(x>0)),4))
 if h==10:
  print('regimes',{int(y):round(float(np.mean([v for v,d in zip(x,dates) if d.year==y])),6) for y in sorted(set(d.year for d in dates))})
q=f.rank(axis=1,pct=True); print('turnover',round(float(q.diff().abs().mean(axis=1).dropna().mean()),6))
