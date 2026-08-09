import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
sig={}
for a,x in p.items():
 c=x.close; r20=c.pct_change(20)
 peak=c.rolling(60).max(); dd=(c/peak-1).rolling(60).min().abs()
 sig[a]=r20/(0.02+dd)
idx=sorted(set().union(*[set(x.index) for x in p.values()])); rows=[]
for d in idx:
 for h in [1,5,10]:
  f=[];y=[]
  for a,x in p.items():
   if d not in x.index: continue
   i=x.index.get_loc(d); z=sig[a].get(d,np.nan)
   if i+h<len(x) and np.isfinite(z): f.append(z); y.append(x.close.iloc[i+h]/x.close.iloc[i]-1)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=df[df.h==h]; print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
s=pd.DataFrame([(d,a,sig[a].get(d,np.nan)) for d in idx for a in A],columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal')
print('turnover_proxy',round(s.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
s.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').to_csv('../persistent/factor_signals_miner_3_20270225_drawdown_quality.csv',index=False)
