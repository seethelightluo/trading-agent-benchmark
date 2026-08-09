import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-24')
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index().loc[:cut] for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); disp=r.sub(r.median(axis=1),axis=0).abs().mean(axis=1)
rev=pd.DataFrame({a:-(p[a]/p[a].shift(3)-1) for a in A}); state=(disp>disp.rolling(60,min_periods=30).median()).astype(float); raw=rev.mul(state,axis=0)
rows=[];obs=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 v={a:raw.at[dt,a] if dt in raw.index else np.nan for a in A}; med=np.nanmedian([z for z in v.values() if np.isfinite(z)])
 for a in A: obs.append((dt,a,v[a]-med if np.isfinite(v[a]) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); z=v[a]-med
   if np.isfinite(z) and i+h<len(p[a]): f.append(z);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8 and len(set(f))>1: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'coverage',round(z.n.mean()/15,4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 z=d[(d.h==1)&(d.date.astype(str)>=lo)&(d.date.astype(str)<=hi)].ic;print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6))
out=pd.DataFrame(obs,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_dispersion_reversal.csv',index=False)
