import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p=Path('../persistent/stock_data')/(a+'.csv'); x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 D[a]=x['close'].loc[:'2027-04-07']
px=pd.DataFrame(D).ffill()
r=px.pct_change()
# lagged 20d relative momentum, scaled by lagged 20d realized volatility
mom=px.pct_change(20).shift(1)
vol=r.rolling(20).std().shift(1)*np.sqrt(252)
f=(mom.div(vol)).sub(mom.div(vol).median(axis=1),axis=0)
# forward close-to-close returns from t to t+ horizons, factor is known at t
out=[]
for h in [1,5,10]:
 fr=px.shift(-h).div(px)-1
 ics=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 s=pd.Series(ics,index=dates).dropna(); out.append((h,len(s),np.mean(ns),s.mean(),s.std(ddof=1),s.mean()/s.std(ddof=1)*np.sqrt(len(s)),(s>0).mean()))
print('period',px.index.min().date(),px.index.max().date(),'assets',len(assets))
for q in out: print('horizon valid_dates avg_n IC ICIR hit',q)
print('coverage',f.notna().sum(axis=1).ge(8).mean(), 'rank_turnover', f.rank(axis=1,pct=True).diff().abs().mean().mean())
for name,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027':('2027','2027-04-07')}.items():
 s=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(px.shift(-1).div(px)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:s.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(name,len(s),np.mean(s) if s else np.nan, (np.mean(s)/np.std(s,ddof=1)*np.sqrt(len(s)) if len(s)>1 else np.nan))
