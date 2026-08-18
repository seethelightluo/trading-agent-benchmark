import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
px=pd.concat(D,axis=1).sort_index(); ret=px.pct_change()
# recovery strength: distance above 60d low, scaled by recent realized volatility; lagged at signal date
low=px.rolling(60,min_periods=45).min(); vol=ret.rolling(20,min_periods=15).std()*np.sqrt(252)
sig=(px/low-1)/vol
# forward 10 trading-day return
fwd=px.shift(-10)/px-1
rows=[]
for dt in sig.index:
 a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_names',r.n.mean(),'coverage',r.n.mean()/15,'10d IC %.5f ICIR %.4f hit %.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for n in [120,252,504,1008]:
 q=r.tail(n); print('recent',n,'dates',len(q),'IC %.5f ICIR %.4f hit %.3f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
# yearly/regime blocks
for y,g in r.groupby(r.index.year): print(y,len(g),'IC %.4f ICIR %.3f'%(g.ic.mean(),g.ic.mean()/g.ic.std()))
# rank turnover 10d sampled
rank=sig.rank(axis=1,pct=True); sampled=rank.iloc[::10]; turn=(sampled.diff().abs().mean(axis=1)).mean(); print('rank_turnover',turn)
sig.to_csv('scripts/miner_2_20350119_recovery_strength_signal.csv')
