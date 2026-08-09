import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allr={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); allr[s]=d.close.pct_change()
R=pd.DataFrame(allr).reindex(columns=U)
# Leave-one-out peer leadership: prior 3-day median return of peers, cross-sectionally comparable
fac=pd.DataFrame(index=R.index,columns=U,dtype=float)
for s in U: fac[s]=R.drop(columns=s).rolling(3,min_periods=2).median().shift(1).mean(axis=1) if False else R.drop(columns=s).rolling(3,min_periods=2).mean().median(axis=1).shift(1)
# Actually peer median at each date, rolling cumulative peer mean then lag one day
# calculate explicitly to avoid lookahead
peer=R.rolling(3,min_periods=2).mean()
for s in U: fac[s]=peer.drop(columns=s).median(axis=1).shift(1)
rows=[]
for s in U:
 d=pd.DataFrame({'date':R.index,'factor':fac[s].values,'y1':R[s].shift(-1).values,'y5':R[s].rolling(5).sum().shift(-5).values,'y10':R[s].rolling(10).sum().shift(-10).values,'symbol':s}); rows.append(d)
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 vals=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   z=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(z): vals.append((dt,z,len(g)))
 z=pd.DataFrame(vals,columns=['date','ic','n']); q=z.ic
 print('H',h,'dates',len(q),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.date.dt.year): print('YR',yr,'dates',len(g),'IC',round(g.ic.mean(),5),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4))
print('coverage',round(x.factor.notna().mean(),4),'turnover',round(x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
x.to_csv('scripts/miner_3_20261217_peer_leadership_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
