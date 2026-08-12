import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
R={s:D[s].close.pct_change() for s in U}
# defensive low-downside-volatility factor; trailing data only, higher means lower downside risk
F={s:-(R[s].clip(upper=0).pow(2).rolling(30,min_periods=20).mean().pow(.5)) for s in U}
for h in [1,5,10]:
  ics=[]; ns=[]; yrs={}
  for dt in sorted(set().union(*[set(x.index) for x in D.values()])):
    xs=[];ys=[]
    for s in U:
      if pd.isna(F[s].get(dt)): continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F[s].loc[dt]); ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic
      if np.isfinite(q): ics.append(q);ns.append(len(xs));yrs.setdefault(str(dt.year),[]).append(q)
  a=np.array(ics); print('h',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(v),4) for k,v in yrs.items()})
FI=pd.concat(F,axis=1); print('coverage',round(np.mean(FI.notna().sum(axis=1)/15),4),'turnover',round(FI.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
for n,p in [('rev5',pd.concat({s:-R[s].rolling(5).sum() for s in U},axis=1)),('mom20',pd.concat({s:R[s].rolling(20).sum() for s in U},axis=1))]: print('corr',n,round(FI.stack().corr(p.stack()),4))
print('window recent',[(y,round(np.mean([z for dt,z in []]),4)) for y in []])
