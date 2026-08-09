import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-08-27')
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for s in U}).sort_index()
R=P.pct_change(fill_method=None)
# compute independently per asset to avoid sparse-calendar rolling contamination
F=pd.DataFrame({s:P[s].pct_change(20,fill_method=None)/(R[s].abs().rolling(20,min_periods=16).sum()) for s in U})
peer=pd.DataFrame({s:P.pct_change(5,fill_method=None).drop(columns=s).median(axis=1) for s in U}); rev=-(P/P.shift(5)-1); mom=P/P.shift(20)-1
print('period',P.index.min().date(),P.index.max().date(),'dates',len(P))
for h in [1,5,10]:
 vals=[]; ns=[]
 for dt in F.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print('H',h,'obs',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4))
print('coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for name,X in [('peer',peer),('rev',rev),('mom',mom)]:
 z=pd.concat([F.stack(),X.stack()],axis=1).dropna(); print('rho',name,round(z.corr(method='spearman').iloc[0,1],5))
for label,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-08-27')]:
 q=[]
 for dt in F.loc[b:a].index:
  z=pd.concat([F.loc[dt],P.shift(-1).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.asarray(q); print('regime',label,'obs',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
