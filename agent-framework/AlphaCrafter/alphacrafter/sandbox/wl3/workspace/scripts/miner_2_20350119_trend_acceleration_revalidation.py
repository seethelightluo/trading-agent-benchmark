import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').sort_index().close.astype(float)
px=pd.concat(P,axis=1).sort_index(); r=px.pct_change(); rv=r.rolling(30,min_periods=20).std()*np.sqrt(20)
sig=(px.pct_change(20)-px.pct_change(60)/3)/rv
fwd=px.shift(-10)/px-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((dt,q,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_names',r.n.mean(),'coverage',r.n.mean()/15,'IC %.5f ICIR %.4f hit %.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for n in [120,252,504]:
 q=r.tail(n); print('recent',n,'dates',len(q),'IC %.5f ICIR %.4f hit %.3f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for y,g in r.groupby(r.index.year): print(y,len(g),'IC %.4f ICIR %.3f'%(g.ic.mean(),g.ic.mean()/g.ic.std()))
print('rank_turnover',sig.rank(axis=1,pct=True).iloc[::10].diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_2_20350119_trend_accel_signal.csv')
