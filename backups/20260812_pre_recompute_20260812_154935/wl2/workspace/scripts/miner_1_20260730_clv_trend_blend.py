import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U})
# Orthogonal interpretable blend: lagged 12-session signed trend breadth plus lagged 1-day CLV.
clv={s:(D[s].close-D[s].low)/(D[s].high-D[s].low).replace(0,np.nan) for s in U}
C=pd.DataFrame(clv).reindex(dates).shift(1)
T=(R.gt(0).rolling(12,min_periods=9).mean()-R.lt(0).rolling(12,min_periods=9).mean()).shift(1)
for w in [0.25,0.5,0.75]:
 F=w*T.rank(axis=1,pct=True)+(1-w)*C.rank(axis=1,pct=True)
 qs=[]; ns=[]; ds=[]
 Y=pd.DataFrame({s:D[s].close.shift(-1).div(D[s].close).sub(1).reindex(dates) for s in U})
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8: qs.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.array(qs); print('weight_trend',w,'dates',len(q),'meanN',round(np.mean(ns),2),'coverage',round(F.notna().sum().sum()/F.size,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'recent252',round(q[-252:].mean(),6),round(q[-252:].mean()/q[-252:].std(ddof=1),6))
 # annual stability
 print('annual',[(y,round(np.mean([q[i] for i,d in enumerate(ds) if d.year==y]),5)) for y in range(2020,2027) if any(d.year==y for d in ds)])
print('done')
