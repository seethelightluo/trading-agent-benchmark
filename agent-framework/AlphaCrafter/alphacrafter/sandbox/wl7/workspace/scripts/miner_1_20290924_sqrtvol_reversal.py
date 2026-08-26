import numpy as np,pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2029-09-24'); b=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(b/(s+'.csv'),parse_dates=['date']).set_index('date').close.rename(s) for s in U],axis=1).sort_index().loc[:end]; R=np.log(P).diff(); vol=R.rolling(20,min_periods=15).std().shift(1)
# Residual short reversal with nonlinear volatility penalty: reversal strength divided by volatility squared-root, avoiding a pure linear duplicate.
x=np.log(P/P.shift(5)); x=x.sub(x.median(axis=1),axis=0); sig=(-x/np.sqrt(vol)).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20]:
 y=np.log(P.shift(-h)/P); vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
 for q,s in [('2025_26',slice('2025','2026')),('2027_28',slice('2027','2028')),('recent',slice('2028-09-01',None))]:
  z=[]
  for d in sig.loc[s].index:
   w=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
   if len(w)>=8:z.append(w.iloc[:,0].corr(w.iloc[:,1]))
  if z: print(q,round(np.nanmean(z),6),round(np.nanmean(z)/np.nanstd(z,ddof=1),6))
print('turnover',round(sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'data_dates',len(P))
sig.stack().rename('signal').to_csv('scripts/miner_1_20290924_sqrtvol_reversal_signal.csv')
