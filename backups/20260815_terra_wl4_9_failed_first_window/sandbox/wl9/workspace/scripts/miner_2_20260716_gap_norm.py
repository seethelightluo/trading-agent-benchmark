import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15'); C={};O={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();d=d[d.index<=end];C[s]=d.close;O[s]=d.open
c=pd.DataFrame(C).sort_index().ffill();o=pd.DataFrame(O).reindex(c.index).ffill(); gap=o/c.shift(1)-1; vol=c.pct_change().rolling(20).std()
for nm,f in [('gap_vol',-gap/vol),('gap_abs',-gap/(gap.abs().rolling(20).mean())),('gap_rank',-gap.rank(axis=1,pct=True))]:
 print('\n',nm)
 for h in [1,5,10]:
  y=c.shift(-h)/c-1;q=[];ns=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  q=pd.Series(q).dropna();print(h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
 print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turn',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
