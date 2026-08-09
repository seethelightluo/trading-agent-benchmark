import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-09-23');D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut,'close'] for s in U};p=pd.DataFrame(D);r=p.pct_change();
for w in [1,2,3,5]:
 a=[]
 for s in U:
  f=-r[s].rolling(w).mean();y=p[s].shift(-1)/p[s]-1
  a += [(dt,s,f.loc[dt],y.loc[dt]) for dt in p.index if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt])]
 a=pd.DataFrame(a,columns=['date','symbol','factor','forward']);z=[];ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1:z.append(spearmanr(g.factor,g.forward).statistic);ns.append(len(g))
 z=np.array(z);print(w,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0))
