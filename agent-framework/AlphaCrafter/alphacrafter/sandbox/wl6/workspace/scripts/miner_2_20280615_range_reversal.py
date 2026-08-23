import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date') for a in A}
cut='2028-06-14'; close=pd.DataFrame({a:D[a]['close'] for a in A}).sort_index().ffill().loc[:cut]; high=pd.DataFrame({a:D[a]['high'] for a in A}).reindex(close.index).ffill(); low=pd.DataFrame({a:D[a]['low'] for a in A}).reindex(close.index).ffill()
r=close.pct_change(); true_range=pd.concat([(high-low)/close, (high-close.shift(1)).abs()/close.shift(1), (low-close.shift(1)).abs()/close.shift(1)],axis=1).values.reshape(len(close),3,len(A)).max(axis=1); true_range=pd.DataFrame(true_range,index=close.index,columns=A)
f=(-r.rolling(3).sum()/true_range.rolling(20).mean()).shift(1)
print('sample',close.index.min().date(),close.index.max().date(),'assets',close.shape[1])
for h in [1,5,10]:
 y=close.shift(-h).div(close)-1; vals=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds).dropna(); print('horizon',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),6),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-25':('2023','2025-12-31'),'2026-27':('2026','2027-12-31'),'2028':('2028','2028-06-14')}.items():
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(close.shift(-1).div(close)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals).dropna(); print('regime',nm,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),6),'hit',round((s>0).mean(),4))
