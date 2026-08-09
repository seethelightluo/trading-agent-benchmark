import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff()
# Low-volatility anomaly: inverse 30-day realized volatility, lagged one day.
# Cross-asset ranking makes scale comparable while preserving a defensive interpretation.
sig=(-r.rolling(30,min_periods=15).std()).shift(1)
y=np.log(p.shift(-1)/p)
print('DATA',p.index.min(),p.index.max(),'assets',len(A))
for h in [1,5,10,20]:
 yy=np.log(p.shift(-h)/p); vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(vals); print('H',h,'dates',len(s),'meanN %.2f'%np.mean(ns),'IC %.6f'%s.mean(),'ICIR %.6f'%(s.mean()/s.std(ddof=1)),'hit %.4f'%np.mean(s>0))
print('coverage',sig.notna().stack().mean(),'mean_valid',sig.notna().sum(axis=1).mean())
ranks=sig.rank(axis=1,pct=True); q=[]
for i in range(10,len(ranks),10):
 z=pd.concat([ranks.iloc[i-10],ranks.iloc[i]],axis=1).dropna();q.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover10',np.mean(q))
for lo,hi in [('2024','2027'),('2028','2030'),('2031','2034')]:
 vals=[]
 for d in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals); print('REG',lo,hi,'dates',len(s),'IC %.6f'%s.mean(),'ICIR %.6f'%(s.mean()/s.std(ddof=1)))
