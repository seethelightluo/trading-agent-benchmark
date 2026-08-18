import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
pdpx=pd.DataFrame(D).sort_index().loc[:'2029-02-07']; r=pdpx.pct_change()
r40=pdpx/pdpx.shift(40)-1; vol=r.rolling(40).std()*np.sqrt(252)
# contrarian direction: negative risk-adjusted intermediate trend, lagged one day
fac=(-(r40/vol)).shift(1)
print('cutoff',pdpx.index.max().date(),'assets',len(U),'dates',len(pdpx))
for h in [1,5,10,20]:
 fr=pdpx.shift(-h)/pdpx-1; a=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(a); print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={np.mean(a):.6f} ICIR={np.mean(a)/np.std(a,ddof=1):.6f} hit={np.mean(a>0):.4f}')
print('coverage',fac.notna().mean().mean(),'avgN',fac.notna().sum(axis=1).mean(),'rank_turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
