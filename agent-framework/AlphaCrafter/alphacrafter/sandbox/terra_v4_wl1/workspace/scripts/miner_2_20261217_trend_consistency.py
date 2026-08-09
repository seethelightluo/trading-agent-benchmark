import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-12-16')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
 P[s]=d[d.index<=cutoff]
p=pd.DataFrame(P); r=p.pct_change(); fac=r.rolling(30,min_periods=24).apply(lambda x:(x>0).mean()-.5,raw=True)
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); q=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 q=np.asarray(q); print('horizon',h,'dates',len(q),'avg_names',round(np.mean(ns),2),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6),'hit',round(np.mean(q>0),4),'coverage',round(np.mean(ns)/15,4))
 if h==1:
  for yr,g in pd.Series(q,index=dates).groupby(lambda x:x.year):print('year',yr,'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),4))
 prev=None; turns=[]
 for dt in p.index:
  z=fac.loc[dt].dropna()
  if len(z)>=8:
   ranks=z.rank(pct=True)
   if prev is not None:
    a=prev.index.intersection(ranks.index)
    if len(a)>=8:turns.append(np.mean(abs(ranks[a]-prev[a])))
   prev=ranks
 print('turnover',round(np.mean(turns),5))
print('period',p.index.min().date(),p.index.max().date(),'assets',len(P))
