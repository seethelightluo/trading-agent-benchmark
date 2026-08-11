import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
F=pd.DataFrame({s:-(d.close/d.open-1) for s,d in D.items()}); P=pd.DataFrame({s:d.close for s,d in D.items()})
for h in [1,5,10]:
 ic=[]; ns=[]
 for i,dt in enumerate(F.index):
  if i+h>=len(P.index): break
  z=pd.concat([F.loc[dt],(P.shift(-h).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(ic); print('horizon',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 a=[]
 for dt in F.index:
  if lo<=dt.year<=hi:
   z=pd.concat([F.loc[dt],(P.shift(-1).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',lo,hi,'dates',len(a),'IC',np.mean(a))
r=F.rank(axis=1,pct=True); print('coverage',F.notna().sum().sum()/(F.shape[0]*15),'turnover', (r.diff().abs().mean(axis=1)*2).mean())

if __name__=='__main__': pass
