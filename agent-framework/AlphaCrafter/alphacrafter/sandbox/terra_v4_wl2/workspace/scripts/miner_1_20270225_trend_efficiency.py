import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
# Trend efficiency: net 40-session return divided by path length, lagged and cross-sectionally demeaned.
raw={}
for a,x in p.items():
 r=x.pct_change(); net=x/x.shift(40)-1; path=r.abs().rolling(40,min_periods=32).sum(); raw[a]=net/(path+0.05)
rows=[]; artifact=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:raw[a].get(dt,np.nan) for a in A}; med=np.nanmedian([v for v in vals.values() if np.isfinite(v)]) if sum(np.isfinite(list(vals.values())))>=8 else np.nan
 for h in [1,5,10]:
  fac=[]; fwd=[]
  for a,x in p.items():
   if dt not in x.index: continue
   i=x.index.get_loc(dt); f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if np.isfinite(f) and i+h<len(x): fac.append(f); fwd.append(x.iloc[i+h]/x.iloc[i]-1)
  if len(fac)>=8: rows.append((dt,h,spearmanr(fac,fwd).statistic,len(fac)))
 for a in A:
  f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
  if np.isfinite(f): artifact.append((dt,a,f))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('horizon',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
out=pd.DataFrame(artifact,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_1_20270225_trend_efficiency.csv',index=False)
r=out.pivot(index='date',columns='asset',values='signal').rank(pct=True,axis=1);print('artifact_rows',len(out),'coverage',len(out)/(len(d[d.h==1])*15),'turnover',r.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-12-31')]:
 z=d[(d.h==1)&(d.date.between(lo,hi))];print('regime',lo,hi,'dates',len(z),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std() if len(z)>1 else np.nan)
