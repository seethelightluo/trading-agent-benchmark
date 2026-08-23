import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-03-06'); Hs=[1,3,5,10]
cl=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:END]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(cl.index).ffill()
panic=((vix.shift(1)>vix.shift(1).rolling(60).median()) | (vix.shift(1)/vix.shift(6)-1>0.05))
vol=cl.pct_change().rolling(20).std().shift(1)
sig=(-(cl/cl.shift(5)-1).shift(1)/vol)
sig=sig.mul(panic.astype(float),axis=0)
sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for h in Hs:
 fr=cl.shift(-h)/cl-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(x): rr.append((dt,x,len(z)))
 D=pd.DataFrame(rr,columns=['date','ic','n']).set_index('date')
 for lab,S in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
  q=S.ic; print('H',h,lab,'dates',len(q),'avg_n',round(S.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 print('---')
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'panic_days',int(panic.sum()),'period',sig.index.min().date(),sig.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290308_vix_panic_reversal_signal.csv',index=False)
