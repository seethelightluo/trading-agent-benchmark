import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).ffill().loc[:'2027-06-16']; r=p.pct_change()
# Lagged medium-vs-long momentum slope: recent 20d return minus preceding 40d return scaled to same length.
f=(r.rolling(20).sum()-r.shift(20).rolling(40).sum()/2).shift(1)
print('period',p.index.min().date(),p.index.max().date(),'assets',len(A))
for h in [1,5,10,20]:
 vals=[]; ns=[]
 fr=p.shift(-h).div(p)-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(vals).dropna(); print('horizon',h,'valid_dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),5),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027YTD':('2027','2027-06-16')}.items():
 x=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(p.shift(-10).div(p)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',nm,'dates',len(x),'IC',round(np.mean(x),5) if x else 'nan','ICIR',round(np.mean(x)/np.std(x,ddof=1)*np.sqrt(len(x)),5) if len(x)>1 else 'nan')
