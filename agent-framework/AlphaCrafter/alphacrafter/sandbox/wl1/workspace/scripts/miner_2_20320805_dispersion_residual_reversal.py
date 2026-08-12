import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C=pd.DataFrame({s:get_stock_daily_data(s,days=3200).set_index('date')['close'] for s in U}).sort_index().ffill(); R=C.pct_change(); bench=R.mean(axis=1)
# residual short-term reversal, gated by unusually high cross-asset dispersion (a regime where reversal may be rewarded)
res=R.sub(bench,axis=0); disp=R.std(axis=1); gate=disp>disp.rolling(120,min_periods=60).median()
f=-res.rolling(5).sum().div(res.rolling(20).std())*gate.astype(float).values[:,None]
rows=[]
for h in [1,5,10,20]:
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],C.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna(); rows.append((h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('rows',len(C),'assets',len(C.columns),'active',gate.mean())
for x in rows: print(x)
h=10; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],C.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); print('coverage',a.n.mean()/15)
for lo,hi in [('2024','2026'),('2027','2029'),('2030','2032')]:
 q=a.loc[lo:hi,'ic'];print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.loc[a.index].to_csv('scripts/miner_2_20320805_dispersion_residual_reversal_signal.csv',index_label='date')
