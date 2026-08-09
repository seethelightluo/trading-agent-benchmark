import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
# Contrarian intraday reversal: yesterday's open-to-close loss receives a positive score.
F={s:(D[s].open/D[s].close-1).replace([np.inf,-np.inf],np.nan) for s in U}; FI=pd.concat(F,axis=1); R={s:D[s].close.pct_change() for s in U}
for h in [1,5,10]:
  ics=[]; ns=[]; years={}
  for dt in FI.index:
    xs=[];ys=[]
    for s in U:
      if dt not in D[s].index or pd.isna(F[s].get(dt)): continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F[s].loc[dt]);ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      z=spearmanr(xs,ys).statistic; ics.append(z); ns.append(len(xs)); years.setdefault(dt.year,[]).append(z)
  a=np.asarray(ics); print('h',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(float(np.mean(v)),4) for k,v in years.items()})
print('coverage',round(FI.notna().sum().sum()/(len(FI)*15),4)); q=FI.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean(axis=1).mean()*2,4))
for name,p in [('rev5',pd.concat({s:-R[s].rolling(5).sum() for s in U},axis=1)),('mom20',pd.concat({s:R[s].rolling(20).sum() for s in U},axis=1)),('clv',pd.concat({s:(D[s].close-D[s].low)/(D[s].high-D[s].low+1e-12) for s in U},axis=1))]: print('corr',name,round(FI.stack().corr(p.stack()),4))
