import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end].close.pct_change()
for w in [20,40,90,120]:
 F=[]
 for s in U:
  r=D[s].close.pct_change(); z=pd.concat([r.rename('r'),v.rename('v')],axis=1).dropna()
  F.append((-z.r.rolling(w,min_periods=max(15,w-15)).cov(z.v)/z.v.rolling(w,min_periods=max(15,w-15)).var()).rename(s))
 F=pd.concat(F,axis=1); out=[]; ns=[]
 for dt,row in F.iterrows():
  xs=[];ys=[]
  for s in U:
   if pd.isna(row[s]) or dt not in D[s].index: continue
   ix=D[s].index.get_loc(dt)
   if ix+1<len(D[s]): xs.append(row[s]);ys.append(D[s].close.iloc[ix+1]/D[s].close.iloc[ix]-1)
  if len(xs)>=8 and len(set(xs))>1: out.append(spearmanr(xs,ys).statistic);ns.append(len(xs))
 a=np.array(out); print('window',w,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(F.notna().sum(axis=1).mean()/15,4))
