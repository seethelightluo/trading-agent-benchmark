import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-03-20'); Hs=[1,3,5,10]
A={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:END] for s in U}
# Close-location pressure: lagged rolling mean of (close-low)/(high-low), centered around 0.5.
cl=pd.DataFrame({s:x.close for s,x in A.items()}).sort_index(); hi=pd.DataFrame({s:x.high for s,x in A.items()}).reindex(cl.index); lo=pd.DataFrame({s:x.low for s,x in A.items()}).reindex(cl.index)
clv=((cl-lo)/(hi-lo).replace(0,np.nan)-.5).rolling(10,min_periods=7).mean().shift(1)
for H in Hs:
 fr=cl.shift(-H)/cl-1; rows=[]
 for dt in clv.index:
  z=pd.concat([clv.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(ic): rows.append((dt,ic,len(z)))
 D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 for label,sub in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
  a=sub.ic; print('H',H,label,'dates',len(a),'avg_n',round(sub.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(clv.notna().sum(axis=1).mean()/15,4),'period',clv.index.min().date(),clv.index.max().date())
out=clv.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290322_clv10_signal.csv',index=False)
