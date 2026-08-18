import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end].close
# defensive VIX beta: negative rolling covariance / VIX variance, computed on exact date intersection
rows=[]
for s in U:
 r=D[s].close.pct_change(); z=pd.concat([r.rename('r'),v.pct_change().rename('v')],axis=1).dropna()
 cov=z.r.rolling(60,min_periods=45).cov(z.v); vv=z.v.rolling(60,min_periods=45).var()
 b=(-cov/vv).rename(s); rows.append(b)
F=pd.concat(rows,axis=1).sort_index()
# evaluate using each asset's next valid observation, avoiding calendar misalignment
for h in [1,5,10]:
  ics=[]; ns=[]; yrs={}
  for dt in F.index:
    xs=[];ys=[]
    for s in U:
      if dt not in D[s].index or pd.isna(F.loc[dt,s]): continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F.loc[dt,s]);ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic; ics.append(q);ns.append(len(xs)); yrs.setdefault(str(dt.year),[]).append(q)
  a=np.array(ics); print('h',h,'N',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'years',{k:round(np.mean(x),4) for k,x in yrs.items()})
print('coverage',np.mean(F.notna().sum(axis=1)/15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2)
print('pooled corr rev5',F.stack().corr((-pd.concat({s:D[s].close.pct_change(5) for s in U},axis=1)).stack()),'mom20',F.stack().corr(pd.concat({s:D[s].close.pct_change(20) for s in U},axis=1).stack()))
