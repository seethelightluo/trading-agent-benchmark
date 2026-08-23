import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=10
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close']
cl=pd.DataFrame(px).sort_index(); r=cl.pct_change(); net=cl/cl.shift(20)-1; path=r.abs().rolling(20).sum(); eff=(net.abs()/path).clip(0,1)
sig=(net*eff).replace([np.inf,-np.inf],np.nan); fr=cl.shift(-H)/cl-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic): rows.append((dt,ic,len(z)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,sub in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
 a=sub.ic; print(label,'dates',len(a),'avg_n',round(sub.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),4),'dates',len(D),'period',D.index.min().date(),D.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290208_efficiency_momentum_signal.csv',index=False)
