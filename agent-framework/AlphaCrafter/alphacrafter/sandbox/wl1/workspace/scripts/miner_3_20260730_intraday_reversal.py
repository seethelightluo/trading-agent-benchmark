import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index() for s in U}
idx=pd.Index(sorted(set.intersection(*[set(x.index) for x in D.values()])))
o=pd.DataFrame({s:D[s].reindex(idx).open for s in U}); c=pd.DataFrame({s:D[s].reindex(idx).close for s in U})
# intraday reversal: prior completed open-to-close move, cross-sectional rank; predicts next close-to-close
f=o/c*0+ (c/o-1); y=c.shift(-1)/c-1
ics=[]; ns=[]
for dt in idx[:-1]:
 q=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1:ics.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
a=np.array(ics); print('assets',len(U),'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in [5,10]:
 yy=c.shift(-h)/c-1; z=[]
 for i,dt in enumerate(idx[:-h]):
  q=pd.concat([f.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(spearmanr(q.f,q.y).statistic)
 z=np.array(z);print('h',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('regime',{int(y):round(pd.Series(ics,index=idx[-len(ics):]).groupby(lambda x:x.year).mean().get(y,np.nan),5) for y in sorted(idx.year.unique())})
