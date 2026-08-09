import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-16')
dat={}; fac={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@end').sort_values('date').set_index('date'); r=d.close.pct_change(); fac[s]=(d.close.pct_change(20)/(r.abs().rolling(20).sum())*np.sign(d.close.pct_change(5))); dat[s]=d.close
for h in [1,5,10]:
 rows=[]
 dates=sorted(set().union(*[x.index for x in dat.values()]))
 for dt in dates:
  vals=[]
  for s in U:
   if dt in fac[s].index:
    f=fac[s].loc[dt]; c=dat[s]; ix=c.index.get_loc(dt)
    if pd.notna(f) and ix+h<len(c):
     y=c.iloc[ix+h]/c.iloc[ix]-1
     if pd.notna(y): vals.append((f,y))
  if len(vals)>=8: rows.append(spearmanr(*zip(*vals)).statistic)
 z=pd.Series(rows); print('horizon',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
# daily detail
rows=[]
for dt in dates:
 vals=[]
 for s in U:
  if dt in fac[s].index:
   c=dat[s]; ix=c.index.get_loc(dt); f=fac[s].loc[dt]
   if pd.notna(f) and ix+1<len(c) and pd.notna(c.iloc[ix+1]): vals.append((f,c.iloc[ix+1]/c.iloc[ix]-1))
 if len(vals)>=8: rows.append((dt,spearmanr(*zip(*vals)).statistic,len(vals)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print(q.groupby(q.index.year).ic.agg(['mean','std']).assign(icir=lambda x:x['mean']/x['std']).round(4)); print('turnover',pd.DataFrame({s:fac[s] for s in U}).reindex(q.index).rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
