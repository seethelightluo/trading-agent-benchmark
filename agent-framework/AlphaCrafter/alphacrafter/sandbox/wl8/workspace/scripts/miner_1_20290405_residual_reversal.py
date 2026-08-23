import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-03-20')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:END] for s in U}
cl=pd.DataFrame(P).sort_index(); r=cl.pct_change(); m=r.mean(axis=1)
# Residual 20-session reversal: asset return less contemporaneous cross-asset benchmark, lagged one day,
# scaled by idiosyncratic rolling volatility. Designed to separate broad beta shocks from relative reversal.
res=r.sub(m,axis=0); vol=res.rolling(20,min_periods=12).std().shift(1)
sig=(-(res.rolling(10,min_periods=8).sum()/vol).shift(1)).clip(-8,8)
fr={h:cl.shift(-h)/cl-1 for h in [5,10]}
for h in [5,10]:
 rows=[]; f=fr[h]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): rows.append((dt,q,len(z)))
 D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 for label,sub in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
  a=sub.ic; print(h,label,'dates',len(a),'avg_n',round(sub.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'period',D.index.min().date(),D.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290405_residual_reversal_signal.csv',index=False)
