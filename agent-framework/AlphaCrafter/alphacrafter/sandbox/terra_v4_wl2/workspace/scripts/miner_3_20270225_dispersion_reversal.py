import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=px.pct_change(); disp=r.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).quantile(.75)).shift(1)
f=(-r.rolling(3).sum()).where(gate); f=f.sub(f.median(axis=1),axis=0); fw={h:px.shift(-h)/px-1 for h in [1,5,10]}
for h in [1,5,10]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 z=pd.Series(vals); print('std',z.std())
for label,start,end in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 vals=[]
 for dt in f.index:
  if str(dt)<start or str(dt)>end: continue
  z=pd.concat([f.loc[dt],fw[1].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(label,len(vals),np.mean(vals) if vals else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_dispersion_reversal.csv',index=False);print('active',gate.sum(),'artifact',len(out))
# rank turnover
q=f.rank(axis=1,pct=True).diff().abs().mean(axis=1);print('turnover',q.mean())
