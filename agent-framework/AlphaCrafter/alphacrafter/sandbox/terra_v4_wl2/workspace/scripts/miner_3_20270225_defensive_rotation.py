import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
r={a:p[a].pct_change() for a in A}; eq=pd.concat([r[a] for a in E],axis=1).median(axis=1)
# Defensive rotation: asset 20d return relative to equity-basket return, scaled by its 20d downside deviation.
raw={}
for a in A:
 rel=(p[a]/p[a].shift(20)-1)-(eq.rolling(20).sum())
 down=r[a].where(r[a]<0,0).rolling(20,min_periods=15).std()
 raw[a]=rel/(down*np.sqrt(20)+0.02)
rows=[]; obs=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 v={a:raw[a].get(dt,np.nan) for a in A}; good=[x for x in v.values() if np.isfinite(x)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: obs.append((dt,a,v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  fac=[]; fw=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); f=v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan
   if np.isfinite(f) and i+h<len(p[a]): fac.append(f);fw.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(fac)>=8: rows.append((dt,h,spearmanr(fac,fw).statistic,len(fac)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(obs,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_defensive_rotation.csv',index=False)
print('artifact',len(out),'turnover',round(out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
# library correlation unavailable without loading all artifacts
print('max_abs_library_correlation',None)
