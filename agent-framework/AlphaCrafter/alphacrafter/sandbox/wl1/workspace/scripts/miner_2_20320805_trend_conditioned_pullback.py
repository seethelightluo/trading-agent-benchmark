import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3200) for s in U}
close=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=close.pct_change()
# Candidate: trend-conditioned, volatility-scaled short pullback: favor recent pullbacks only when medium trend is positive.
trend=close.pct_change(60)
vol=r.rolling(20).std()*np.sqrt(252)
r5=close.pct_change(5)
f=(-r5/vol)*(trend>0).astype(float)
# damp extreme signals cross-sectionally only via rank (preserves interpretation)
fr=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,5,10,20]:
  ic=[]
  for dt in fr.index:
    x=fr.loc[dt]; y=close.pct_change(h).shift(-h).loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  a=pd.Series(ic).dropna()
  rows.append((h,len(a),a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()))
print('rows',len(close),'assets',len(close.columns),'factor_dates',fr.dropna(how='all').shape[0])
for x in rows: print('horizon dates IC ICIR hit',x)
# admission same horizon 10d, turnover, coverage, regimes
h=10; vals=[]
for dt in fr.index:
 x=fr.loc[dt]; y=close.pct_change(h).shift(-h).loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
print('coverage',a.n.mean()/15,'regimes')
for lo,hi in [('2024','2026'),('2027','2029'),('2030','2032')]:
 q=a.loc[lo:hi,'ic']; print(lo, len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# rank turnover (cross-sectional rank changes)
ranks=fr.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).dropna(); print('turnover',turn.mean())
# save reproducible signal artifact
out=fr.loc[a.index].copy(); out.to_csv('scripts/miner_2_20320805_trend_conditioned_pullback_signal.csv',index_label='date')
