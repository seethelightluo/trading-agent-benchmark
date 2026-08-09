import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
C={}; O={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=end]; C[s]=d.close; O[s]=d.open
c=pd.DataFrame(C).sort_index().ffill(); o=pd.DataFrame(O).reindex(c.index).ffill()
g=o/c.shift(1)-1
# gap mean reversion; exclude same-day close return information
for name,f in [('gap1',-g),('gap3',-g.rolling(3).mean()),('gap5',-g.rolling(5).mean()),('gap_z',-(g-g.rolling(20).mean())/g.rolling(20).std())]:
 print('\n',name)
 for h in [1,5,10]:
  y=c.shift(-h)/c-1; q=[]; ns=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  q=pd.Series(q).dropna(); print(h,'N',len(q),'meanN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 rank=f.rank(axis=1,pct=True); print('turn',((rank-rank.shift()).abs().mean(axis=1)).mean(),'coverage',f.notna().sum(axis=1).ge(8).mean())
 # correlations against proxies
 for nm,x in [('mom20',c/c.shift(20)-1),('rev5',-(c/c.shift(5)-1))]:
  a=pd.concat([f.stack(),x.stack()],axis=1).dropna();print('corr',nm,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
