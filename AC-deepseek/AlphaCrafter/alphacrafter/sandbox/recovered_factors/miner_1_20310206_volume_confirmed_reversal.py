import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
d={}
for a in assets:
 p=f'{base}/{a}.csv'
 x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 d[a]=x
idx=pd.date_range('2020-01-01','2031-02-05',freq='D')
close=pd.DataFrame({a:d[a]['close'].reindex(idx).ffill() for a in assets})
vol=pd.DataFrame({a:d[a]['volume'].reindex(idx).fillna(0) for a in assets})
r=close.pct_change()
# Candidate: volume-confirmed short-term reversal, scaled by volatility.
# High abnormal volume makes a recent selloff more informative as capitulation/rebound potential.
ls=np.log1p(vol)
vs=(ls-ls.rolling(20,min_periods=10).mean())/ls.rolling(20,min_periods=10).std()
ret5=close.pct_change(5)
rv=r.rolling(20,min_periods=10).std()
f=-(ret5/rv)*vs
# cross-sectional daily rank-like demean
f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for h in [1,5,10,20]:
  ic=[]; ns=[]
  fr=close.pct_change(h).shift(-h)
  for t in f.index:
   z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
   if len(z)>=8:
    ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  arr=np.array(ic); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(arr),np.nanmean(arr)/np.nanstd(arr,ddof=1),np.mean(arr>0),len(arr),np.mean(ns)))
# turnover 10d via cross-sectional ranks
q=f.rank(axis=1,pct=True); print('turnover10',np.nanmean((q-q.shift(10)).abs().mean(axis=1)))
print('cells',f.notna().sum().sum(),'total',f.size,'coverage',f.notna().mean())
# regimes
fr=close.pct_change(10).shift(-10)
for y in [2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031]:
 ic=[]
 for t in f.index[f.index.year==y]:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if ic: print(y,len(ic),np.mean(ic),np.mean(ic)/np.std(ic,ddof=1) if len(ic)>1 else np.nan)
print('last date',f.index[-1])
