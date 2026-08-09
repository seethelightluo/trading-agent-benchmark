import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in assets}
r={a:p[a].pct_change() for a in assets}; m=r['SPX']
# Market-neutral medium-term momentum: trailing 20d asset return less rolling 60d beta to SPX times SPX return.
# beta and signal at date t use observations through t; forward return starts t+1.
beta={}; sigraw={}
for a in assets:
 cov=r[a].rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var()
 beta[a]=cov/(var+1e-12)
 sigraw[a]=(p[a]/p[a].shift(20)-1)-beta[a]*(p['SPX']/p['SPX'].shift(20)-1)
idx=sorted(set().union(*[set(x.index) for x in p.values()])); rows=[]; artifact=[]
for dt in idx:
 vals={a:sigraw[a].get(dt,np.nan) for a in assets}; good=[v for v in vals.values() if np.isfinite(v)]
 med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in assets: artifact.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  fac=[]; fwd=[]
  for a in assets:
   if dt not in p[a].index: continue
   ix=p[a].index.get_loc(dt); f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if ix+h<len(p[a]) and np.isfinite(f):
    y=p[a].iloc[ix+h]/p[a].iloc[ix]-1
    if np.isfinite(y): fac.append(f); fwd.append(y)
  if len(fac)>=8: rows.append((dt,h,spearmanr(fac,fwd).statistic,len(fac)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else None)
out=pd.DataFrame(artifact,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_idio_mom.csv',index=False)
wide=out.pivot(index='date',columns='asset',values='signal'); print('turnover',round(wide.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
print('artifact',len(out))
