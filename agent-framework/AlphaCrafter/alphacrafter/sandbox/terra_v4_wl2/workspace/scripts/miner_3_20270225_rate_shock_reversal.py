import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); r3=pd.DataFrame({a:p[a].pct_change(3) for a in A}); base=r3.median(axis=1); resid=r3.sub(base,axis=0)
rate=np.log(p['US10Y']).diff(5); threshold=rate.rolling(252,min_periods=60).quantile(.75); active=rate>threshold
rows=[];sig=[]
for dt in r.index:
 if not bool(active.get(dt,False)): continue
 vals=-resid.loc[dt]; good=vals.dropna(); med=good.median() if len(good)>=8 else np.nan
 for a in A: sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if not np.isfinite(vals.get(a,np.nan)) or not np.isfinite(med) or dt not in p[a].index: continue
   ix=p[a].index.get_loc(dt)
   if ix+h<len(p[a]): f.append(vals[a]-med);y.append(p[a].iloc[ix+h]/p[a].iloc[ix]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('H',h,'dates',len(z),'avg_n',z.n.mean() if len(z) else 0,'IC',z.ic.mean() if len(z) else np.nan,'ICIR',z.ic.mean()/z.ic.std() if len(z)>1 else np.nan,'hit',(z.ic>0).mean() if len(z) else 0)
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_rate_shock_reversal.csv',index=False);print('artifact',len(out))
