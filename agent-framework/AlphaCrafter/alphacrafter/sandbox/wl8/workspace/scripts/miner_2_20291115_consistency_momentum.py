import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.astype(float)
p=pd.DataFrame(P).loc[:'2029-11-14']; r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Volatility-normalized medium-term momentum, with a lagged trend-consistency multiplier.
base=p.pct_change(20).shift(1)/vol.shift(1)
cons=(np.sign(r.shift(1)).rolling(20,min_periods=15).mean()).clip(-1,1)
sig=base*(0.5+0.5*cons)
f=p.shift(-10)/p-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('factor=consistency-weighted 20d vol-adjusted momentum')
print('dates',len(df),'avg_n',df.n.mean(),'coverage',df.n.mean()/15,'IC10',df.ic.mean(),'ICIR10',df.ic.mean()/df.ic.std(ddof=1),'hit', (df.ic>0).mean())
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a);print('decay',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
for lo,hi in [('2020-01-01','2023-12-31'),('2024-01-01','2026-07-15'),('2026-07-16','2027-12-31'),('2028-01-01','2029-11-14')]:
 x=df.loc[lo:hi].ic
 print('regime',lo,hi,'dates',len(x),'IC',x.mean() if len(x) else np.nan,'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
q=sig.rank(pct=True); turnover=q.diff().abs().mean(axis=1).dropna().mean(); print('turnover',turnover)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291115_consistency_momentum_signal.csv',index=False)
