import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
r=pd.concat([p[a].pct_change().rename(a) for a in A],axis=1)
spx=p['SPX']; trend=(spx.shift(1)<spx.shift(21)); breadth=(r.gt(0).sum(axis=1)/r.notna().sum(axis=1)).shift(1)<=.40
base={a:-(p[a].shift(1)/p[a].shift(4)-1) for a in A}; rows=[]; sig=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 if not bool(trend.get(dt,False) and breadth.get(dt,False)): continue
 vals={a:base[a].get(dt,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for h in [1,5,10]:
  fac=[];fw=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt);f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if np.isfinite(f) and i+h<len(p[a]):fac.append(f);fw.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(fac)>=8: rows.append((dt,h,spearmanr(fac,fw).statistic,len(fac)))
 for a in A:sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h].dropna();print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_riskoff_reversal.csv',index=False);print('artifact',len(out),'max_abs_library_correlation',None)
