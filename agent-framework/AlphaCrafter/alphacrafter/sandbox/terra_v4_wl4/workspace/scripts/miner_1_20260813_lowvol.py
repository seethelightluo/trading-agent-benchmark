import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d['r']=d.close.pct_change(); D[s]=d.set_index('date')
# trailing low-volatility signal, lower realized vol ranks higher
rows=[]
for dt in sorted(set.intersection(*[set(x.index) for x in D.values()])):
 vals={}; fw={}
 for s,d in D.items():
  if dt not in d.index: continue
  ix=d.index.get_loc(dt)
  if ix<20 or ix+1>=len(d): continue
  vals[s]=-d.r.iloc[ix-19:ix+1].std()
  fw[s]=d.r.iloc[ix+1]
 if len(vals)>=8:
  a=pd.Series(vals); b=pd.Series(fw).reindex(a.index)
  rows.append((dt,spearmanr(a,b).statistic))
x=pd.DataFrame(rows,columns=['date','ic']).dropna()
print('dates',len(x),'mean_n',15,'IC %.5f ICIR %.5f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=x[(x.date>=lo)&(x.date<=hi+'-12-31')]; print(lo,hi,len(z),z.ic.mean(),z.ic.mean()/z.ic.std())
# horizons
for h in [5,10]:
 rows=[]
 for dt in sorted(set.intersection(*[set(x.index) for x in D.values()])):
  vals={}; fw={}
  for s,d in D.items():
   if dt not in d.index: continue
   ix=d.index.get_loc(dt)
   if ix<20 or ix+h>=len(d): continue
   vals[s]=-d.r.iloc[ix-19:ix+1].std(); fw[s]=d.close.iloc[ix+h]/d.close.iloc[ix]-1
  if len(vals)>=8: rows.append(spearmanr(pd.Series(vals),pd.Series(fw).reindex(vals)).statistic)
 z=pd.Series(rows).dropna(); print(h,'IC %.5f ICIR %.5f n'% (z.mean(),z.mean()/z.std()),len(z))
