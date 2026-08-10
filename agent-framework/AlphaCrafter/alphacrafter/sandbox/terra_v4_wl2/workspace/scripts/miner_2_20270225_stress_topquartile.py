import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}; r=pd.DataFrame({a:p[a].pct_change() for a in A}); r3=r.rolling(3).sum(); resid=r3-r3.median(axis=1); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close.reindex(r.index).ffill(); stress=v.shift(1)>v.shift(1).rolling(120,min_periods=60).quantile(.75)
rows=[]; sig=[]
for dt in r.index:
 if not stress.get(dt,False): continue
 vals=-resid.loc[dt]; good=vals.dropna()
 if len(good)<8: continue
 med=good.median()
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if not np.isfinite(vals.get(a,np.nan)) or dt not in p[a].index: continue
   i=p[a].index.get_loc(dt)
   if i+h<len(p[a]): f.append(vals[a]-med);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
 for a in A:sig.append((dt,a,vals.get(a,np.nan)-med if np.isfinite(vals.get(a,np.nan)) else np.nan))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_stress_topquartile.csv',index=False);print('coverage',out.signal.notna().mean(),'turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
