import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
F={}; R={}
for s in U:
 r=D[s].close.pct_change(); R[s]=r
 up=r.clip(lower=0).rolling(20,min_periods=15).mean(); dn=(-r.clip(upper=0)).rolling(20,min_periods=15).mean()
 F[s]=(up/(dn+1e-8)).replace([np.inf,-np.inf],np.nan)
FI=pd.concat(F,axis=1)
for h in [1,5,10]:
  ics=[]; ns=[]; yrs={}
  for dt in FI.index:
    xs=[];ys=[]
    for s in U:
      if dt not in D[s].index or pd.isna(F[s].get(dt)): continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F[s].loc[dt]);ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic; ics.append(q); ns.append(len(xs)); yrs.setdefault(str(dt.year),[]).append(q)
  a=np.array(ics); print('h',h,'N',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(x),4) for k,x in yrs.items()})
print('coverage',round(np.mean(FI.notna().sum(axis=1)/15),4),'turnover',round(FI.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
for n,p in [('rev5',pd.concat({s:-R[s].rolling(5).sum() for s in U},axis=1)),('mom20',pd.concat({s:R[s].rolling(20).sum() for s in U},axis=1)),('clv',pd.concat({s:(D[s].close-D[s].low)/(D[s].high-D[s].low+1e-12) for s in U},axis=1))]: print('corr',n,FI.stack().corr(p.stack()))
