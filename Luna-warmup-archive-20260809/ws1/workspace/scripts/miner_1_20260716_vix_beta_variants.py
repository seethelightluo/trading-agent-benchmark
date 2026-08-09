import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end].close
vp=v.pct_change(); vz=(v-v.rolling(252,min_periods=100).mean())/v.rolling(252,min_periods=100).std()
base={}
for s in U:
 z=pd.concat([D[s].close.pct_change().rename('r'),vp.rename('v')],axis=1,join='inner').dropna()
 b=-z.r.rolling(60,min_periods=45).cov(z.v)/z.v.rolling(60,min_periods=45).var()
 base[s]=b
F0=pd.concat(base,axis=1).sort_index()
variants={'beta':F0,'beta_vlevel':F0*vz.reindex(F0.index).clip(lower=0),'beta_absv':F0*vp.abs().rolling(5).mean().reindex(F0.index),'beta_shock':F0*vp.rolling(20).mean().reindex(F0.index),'beta_ranklevel':F0*(1+vz.reindex(F0.index).clip(lower=-1,upper=3))}
for name,F in variants.items():
  for h in [1,5]:
    ic=[]; ns=[]; yrs={}
    for dt,row in F.iterrows():
      xs=[];ys=[]
      for s in U:
       if pd.isna(row.get(s)): continue
       ix=D[s].index.searchsorted(dt)
       if ix>=len(D[s]) or D[s].index[ix]!=dt or ix+h>=len(D[s]): continue
       y=D[s].close.iloc[ix+h]/D[s].close.iloc[ix]-1
       if pd.notna(y): xs.append(row[s]);ys.append(y)
      if len(xs)>=8 and len(set(xs))>1:
       q=spearmanr(xs,ys).statistic;ic.append(q);ns.append(len(xs));yrs.setdefault(str(dt.year),[]).append(q)
    a=np.array(ic); print(name,h,'N',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),3),'years', {k:round(np.mean(x),4) for k,x in yrs.items()})
  print('coverage',round((F.notna().sum(axis=1)/15).mean(),3),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,3))
