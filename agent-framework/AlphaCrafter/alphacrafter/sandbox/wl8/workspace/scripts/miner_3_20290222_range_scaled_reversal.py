import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-02-20'); H=10
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:END] for s in U}; cl=pd.DataFrame(P).sort_index(); r=cl.pct_change();
# Reversal scaled by lagged realized range: penalize noisy assets while preserving short-term mean reversion.
ret5=cl/cl.shift(5)-1; range20=(cl.rolling(20).max()/cl.rolling(20).min()-1).shift(1); sig=(-(ret5.shift(1))/range20).clip(-8,8)
fr=cl.shift(-H)/cl-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic): rows.append((dt,ic,len(z)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,sub in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
 a=sub.ic; print(label,'dates',len(a),'avg_n',round(sub.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'period',D.index.min().date(),D.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290222_range_scaled_reversal_signal.csv',index=False)
