import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p=Path('../persistent/stock_data')/(a+'.csv'); x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); D[a]=x['close']
px=pd.DataFrame(D).ffill(); px=px.loc[:'2027-04-21']; r=px.pct_change()
# Relative 60-session momentum, lagged one day; cross-sectional demeaning avoids market beta.
f=px.pct_change(60).shift(1); f=f.sub(f.median(axis=1),axis=0)
fr={h:px.shift(-h).div(px)-1 for h in [1,5,10,20]}
print('period',px.index.min().date(),px.index.max().date(),'assets',len(assets))
for h in fr:
  vals=[]; ns=[]
  for dt in f.index:
    z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
    if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  s=pd.Series(vals).dropna(); print('horizon',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),5),'hit',round((s>0).mean(),4))
print('coverage_dates',round(f.notna().sum(axis=1).ge(8).mean(),4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for name,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027':('2027','2027-04-21')}.items():
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr[1].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals); print('regime',name,'dates',len(s),'IC',round(s.mean(),5) if len(s) else None,'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),5) if len(s)>1 else None)
