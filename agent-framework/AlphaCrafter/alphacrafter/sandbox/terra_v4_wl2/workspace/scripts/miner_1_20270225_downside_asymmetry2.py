import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r={a:p[a].pct_change() for a in A}
# Downside asymmetry: reward assets with fewer/lower downside shocks relative to total risk.
down={a:r[a].where(r[a]<0,0).rolling(40,min_periods=30).std() for a in A}; tot={a:r[a].rolling(40,min_periods=30).std() for a in A}
raw={a:-(down[a]/(tot[a]+1e-8)) for a in A}
idx=sorted(set().union(*[set(x.index) for x in p.values()])); rows=[]; sig=[]
for d in idx:
 vals={a:raw[a].get(d,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: sig.append((d,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  x=[];y=[]
  for a in A:
   f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if d not in p[a].index or not np.isfinite(f):continue
   i=p[a].index.get_loc(d)
   if i+h<len(p[a]):
    yy=p[a].iloc[i+h]/p[a].iloc[i]-1
    if np.isfinite(yy):x.append(f);y.append(yy)
  if len(x)>=8: rows.append((d,h,spearmanr(x,y).statistic,len(x)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=df[df.h==h]; print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_downside_asymmetry2.csv',index=False)
wide=out.pivot(index='date',columns='asset',values='signal');print('turnover',round(wide.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
