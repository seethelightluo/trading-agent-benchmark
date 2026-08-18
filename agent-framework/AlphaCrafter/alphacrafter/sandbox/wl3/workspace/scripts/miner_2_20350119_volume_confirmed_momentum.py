import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); P[s]=x.close.astype(float); V[s]=x.volume.astype(float)
px=pd.concat(P,axis=1).sort_index(); vol=pd.concat(V,axis=1).reindex(px.index); r=px.pct_change()
# volume-confirmed intermediate momentum: 20d return multiplied by standardized log-volume surprise, lagged
rv=np.log1p(vol).rolling(60,min_periods=40).mean(); rs=np.log1p(vol)-rv; vs=rs.rolling(60,min_periods=40).std(); surprise=rs/vs
sig=px.pct_change(20)*surprise
fwd=px.shift(-10)/px-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_names',r.n.mean(),'coverage',r.n.mean()/15,'IC %.5f ICIR %.4f hit %.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for n in [120,252,504,1008]:
 q=r.tail(n); print('recent',n,'IC %.5f ICIR %.4f hit %.3f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for y,g in r.groupby(r.index.year): print(y,len(g),'IC %.4f ICIR %.3f'%(g.ic.mean(),g.ic.mean()/g.ic.std()))
print('rank_turnover',sig.rank(axis=1,pct=True).iloc[::10].diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_2_20350119_volume_confirmed_momentum_signal.csv')
