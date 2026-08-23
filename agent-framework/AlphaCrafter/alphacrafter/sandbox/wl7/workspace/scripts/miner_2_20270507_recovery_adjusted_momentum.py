import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# candidate: recovery-adjusted medium momentum: 30d return, normalized volatility, penalize current drawdown from 90d high
mom=p.shift(1).pct_change(30)
vol=r.shift(1).rolling(30).std()
high=p.shift(1).rolling(90).max(); dd=p.shift(1)/high-1
sig=(mom/(vol*np.sqrt(30))).where(vol>0) + 0.5*dd
fwd=p.shift(-1)/p-1
rows=[]
for dt in sig.index:
 a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*len(U)))
print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
print('turnover',sig.rank(axis=1).diff().abs().mean(axis=1).mean()/len(U))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 x=q.loc[a:b]; print(a,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1) if len(x)>1 else np.nan)
for h in [5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
q.to_csv('scripts/miner_2_20270507_recovery_adjusted_momentum_ic.csv')
# signal artifact
sig.to_csv('scripts/miner_2_20270507_recovery_adjusted_momentum_signal.csv')
