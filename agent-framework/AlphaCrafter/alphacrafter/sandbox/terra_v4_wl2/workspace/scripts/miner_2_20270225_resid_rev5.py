import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r={a:x.pct_change() for a,x in p.items()}; m=r['SPX']; var=m.rolling(60,min_periods=40).var()
# residual reversal: negate 5-day return unexplained by rolling market beta
sig={}
for a in A:
 beta=r[a].rolling(60,min_periods=40).cov(m)/(var+1e-12)
 sig[a]=-(p[a]/p[a].shift(5)-1-beta*(p['SPX']/p['SPX'].shift(5)-1))
rows=[]; art=[]
for dt in sorted(set().union(*[set(x.index) for x in p.values()])):
 vals={a:sig[a].get(dt,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: art.append((dt,a,vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if dt not in p[a].index: continue
   i=p[a].index.get_loc(dt); v=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
   if np.isfinite(v) and i+h<len(p[a]): f.append(v);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h]; print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4),'cov',round(z.n.mean()/15,4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
out=pd.DataFrame(art,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_2_20270225_resid_rev5.csv',index=False);print('artifact',len(out),'turnover',round(out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
