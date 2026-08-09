import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); r5=r.rolling(5,min_periods=5).sum(); med5=r5.median(axis=1); resid=r5.sub(med5,axis=0)
rate=(r5['US10Y']+r5['CN10Y'])/2; rz=(rate-rate.rolling(60,min_periods=30).mean())/rate.rolling(60,min_periods=30).std(); intensity=((rz.abs()-1.0).clip(lower=0)/2).clip(upper=1)
rows=[]; sig=[]
for dt in r.index:
 vals=(-resid.loc[dt]) if intensity.loc[dt]>0 else (-resid.loc[dt])*np.nan; valid=vals.dropna(); med=np.nanmedian(valid) if len(valid)>=8 else np.nan
 for a in A: sig.append((dt,a, vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if a not in vals or not np.isfinite(vals[a]) or not np.isfinite(med) or dt not in p[a].index: continue
   i=p[a].index.get_loc(dt)
   if i+h<len(p[a]): f.append(vals[a]-med);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8 and np.std(f)>0: rows.append((dt,h,spearmanr(f,y).statistic,len(f),float(intensity.loc[dt])))
d=pd.DataFrame(rows,columns=['date','h','ic','n','intensity'])
for h in [1,5,10]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_rate_shock_resid_reversal.csv',index=False);print('artifact',len(out));print('turnover',round(out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
