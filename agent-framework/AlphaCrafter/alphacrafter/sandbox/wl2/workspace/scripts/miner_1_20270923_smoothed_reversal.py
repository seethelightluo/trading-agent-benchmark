import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in S:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):x=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); P[s]=x.loc[x.index<='2027-09-23']
d=pd.DataFrame({s:x.close for s,x in P.items()}).sort_index(); r=d.pct_change()
# low-turnover short reversal: negative 3-session return, normalized by 20d volatility and smoothed over 3 observations
vol=r.rolling(20,min_periods=15).std(); raw=(-r.rolling(3,min_periods=3).sum()/vol.replace(0,np.nan)).clip(-5,5)
f=raw.rolling(3,min_periods=3).mean().shift(1); y=r.shift(-1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/len(o)/15)
print('IC %.8f ICIR %.8f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()));print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=o.loc[a:b].ic; print(a,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 y=(1+r).rolling(h).apply(np.prod,raw=True).shift(-h); q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
print('max_abs_library_correlation unavailable')
