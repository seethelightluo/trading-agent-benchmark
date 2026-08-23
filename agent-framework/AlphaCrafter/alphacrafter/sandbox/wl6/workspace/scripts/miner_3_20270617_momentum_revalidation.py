import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).ffill().loc[:'2027-06-16']; r=p.pct_change()
f=(r.rolling(20).sum()/(r.rolling(20).std())).shift(1)
for h in [1,5,10]:
 x=p.shift(-h).div(p)-1; vals=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds).dropna(); print('horizon',h,'valid_dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),5),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027':('2027','2027-06-16')}.items():
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(p.shift(-1).div(p)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals).dropna(); print('regime',nm,'dates',len(s),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),5))
