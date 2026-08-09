import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:end,'close'] for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
for w in [20,40,60,120]:
 cov=r.rolling(w,min_periods=max(12,int(w*.7))).cov(m); var=m.rolling(w,min_periods=max(12,int(w*.7))).var(); f=-(cov.div(var,axis=0))
 vals=[]; ns=[]
 for dt in f.index:
  xs=[];ys=[]
  for s in U:
   if pd.isna(f.loc[dt,s]) or dt not in D[s].index: continue
   i=D[s].index.get_loc(dt)
   if i+1<len(D[s]) and pd.notna(D[s].iloc[i+1]): xs.append(f.loc[dt,s]); ys.append(D[s].iloc[i+1]/D[s].iloc[i]-1)
  xa,ya=np.asarray(xs,float),np.asarray(ys,float)
  if len(xa)>=8 and np.isfinite(xa).all() and np.isfinite(ya).all() and len(np.unique(xa))>1 and len(np.unique(ya))>1: vals.append(spearmanr(xa,ya).statistic);ns.append(len(xa))
 a=np.array(vals); print(f'w={w} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/15:.3f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={np.mean(a>0):.4f}')
