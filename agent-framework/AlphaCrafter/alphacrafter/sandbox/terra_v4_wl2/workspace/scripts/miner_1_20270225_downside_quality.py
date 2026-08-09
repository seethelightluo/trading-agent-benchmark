import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
raw={}
for a in A:
 r=p[a].pct_change(); down=r.where(r<0,0.0); ds=down.rolling(60,min_periods=40).std(); raw[a]=p[a].pct_change(60)/(ds+0.01)
rows=[]; obs=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 v={a:raw[a].get(dt,np.nan) for a in A}; g=[x for x in v.values() if np.isfinite(x)]; med=np.nanmedian(g) if len(g)>=8 else np.nan
 for a in A:
  f=v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan
  obs.append((dt,a,f))
 for h in [1,5,10]:
  fac=[];fwd=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); f=v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan
   if np.isfinite(f) and i+h<len(p[a]): fac.append(f);fwd.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(fac)>=8: rows.append((dt,h,spearmanr(fac,fwd).statistic,len(fac)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(obs,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_downside_quality.csv',index=False);print('artifact',len(out),'turnover',out.pivot(index='date',columns='asset',values='signal').rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
