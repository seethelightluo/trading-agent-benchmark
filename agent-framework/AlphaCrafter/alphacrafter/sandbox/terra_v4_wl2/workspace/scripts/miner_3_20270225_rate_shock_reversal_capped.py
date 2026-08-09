import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-02-25')
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').sort_values('date').set_index('date').close for a in A}
r3=pd.DataFrame({a:p[a].pct_change(3) for a in A}); base=r3.median(axis=1); resid=r3.sub(base,axis=0)
rate=np.log(p['US10Y']).diff(5); threshold=rate.rolling(252,min_periods=60).quantile(.75); active=rate>threshold
rows=[];sig=[]
for dt in r3.index:
 if not bool(active.get(dt,False)): continue
 vals=-resid.loc[dt]; good=vals.dropna()
 if len(good)<8: continue
 med=good.median()
 for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if not np.isfinite(vals.get(a,np.nan)): continue
   ix=p[a].index.get_indexer([dt])[0]
   if ix>=0 and ix+h<len(p[a]): f.append(vals[a]-med);y.append(p[a].iloc[ix+h]/p[a].iloc[ix]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('H',h,'dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),q.mean() if len(q) else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_rate_shock_reversal.csv',index=False);print('artifact',len(out))
