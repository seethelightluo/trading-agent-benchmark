import numpy as np,pandas as pd
END=pd.Timestamp('2029-04-19'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=10
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:END]
 # signed intraday pressure, robust to differing scales
 rng=(x.high-x.low).replace(0,np.nan)
 D[s]=((x.close-x.open)/rng).clip(-1,1)
pr={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:END] for s in U}
cl=pd.DataFrame(pr).sort_index(); pressure=pd.DataFrame(D).reindex(cl.index).sort_index()
# 10d accumulated pressure, volatility-normalized and lagged one completed session
ret=cl.pct_change(); vol=ret.rolling(20).std(); sig=(pressure.rolling(10).sum()/vol).shift(1)
# cross-sectional demean to isolate relative pressure
sig=sig.sub(sig.median(axis=1),axis=0)
fr=cl.shift(-H)/cl-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): rows.append((dt,q,len(z)))
DIC=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for lab,s in [('full',DIC),('recent180',DIC.tail(180)),('recent360',DIC.tail(360)),('2026',DIC.loc['2026']),('2027-28',DIC.loc['2027':'2028']),('2029',DIC.loc['2029'])]:
 if len(s):
  a=s.ic; print(lab,'dates',len(a),'avg_n',round(s.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'period',DIC.index.min().date(),DIC.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290419_range_pressure_signal.csv',index=False)
