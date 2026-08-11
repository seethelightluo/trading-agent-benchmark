import numpy as np, pandas as pd
from scipy.stats import spearmanr
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close.astype(float) for s in SYMS}
f={s:p[s].pct_change(5)-p[s].pct_change(20)/4 for s in SYMS}
for h in [1,5,10]:
 out=[]; ns=[]
 for d in sorted(set().union(*[x.index for x in p.values()])):
  vals=[]; ys=[]
  for s in SYMS:
   if d not in p[s].index or pd.isna(f[s].get(d,np.nan)): continue
   ix=p[s].index.get_loc(d)
   if ix+h>=len(p[s]): continue
   vals.append(f[s].loc[d]); ys.append(p[s].iloc[ix+h]/p[s].iloc[ix]-1)
  if len(vals)>=8 and len(set(vals))>1: out.append(spearmanr(vals,ys).statistic); ns.append(len(vals))
 a=np.array(out); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
r=pd.DataFrame(f).rank(axis=1,pct=True); t=[]
for i in range(1,len(r)):
 z=r.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:t.append(np.mean(abs(z.iloc[:,1]-z.iloc[:,0])))
print('turnover',round(float(np.mean(t)),6),'turnover_obs',len(t))
