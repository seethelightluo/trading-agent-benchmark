import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}; dates=D['SPX'].index
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
# Orthogonal-ish risk-adjusted trend: intermediate momentum and 12-session directional consistency, both volatility scaled.
mom=R.rolling(12,min_periods=9).sum()/R.rolling(20,min_periods=15).std()
cons=R.gt(0).rolling(12,min_periods=9).mean()-R.lt(0).rolling(12,min_periods=9).mean()
# rank averaging is robust to cross-asset scale; lag before use
F=(mom.rank(axis=1,pct=True)+cons.rank(axis=1,pct=True))/2
F=F.shift(1)
print('idea rank_blend_risk_adjusted_momentum_consistency; universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1); q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=q[[d.year==yr for d in ds]];print('regime',yr,len(x),round(x.mean(),6) if len(x) else None,round(x.mean()/x.std(ddof=1),4) if len(x)>1 else None)
  for k in [252,504]:
   x=q[-k:];print('recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
print('dates with >=8 are used; no future data')
