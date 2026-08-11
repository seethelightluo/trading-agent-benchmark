import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
vx=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
idx=sorted(set().union(*[set(x.index) for x in P.values()])|set(vx.index)); p=pd.DataFrame({s:x.reindex(idx) for s,x in P.items()}).ffill(); v=vx.reindex(idx).ffill()
r=p.pct_change(); peer=(r.sum(axis=1).to_numpy()[:,None]-r.to_numpy())/(len(U)-1)
# peer leadership: lagged 3-session peer return, rewarded in calm regime and reversed in stress
peer3=pd.DataFrame(peer,index=p.index,columns=U).rolling(3).sum()
vz=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-8)
# simple interpretable conditional: trend-following when VIX below mean, reversal when above mean
f=peer3.mul(np.where(vz.to_numpy()[:,None]>0,-1.0,1.0),axis=0).shift(1)
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: I.append(spearmanr(q.f,q.y).statistic); Ns.append(len(q)); ds.append(p.index[i])
 a=np.array(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('annual10',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
