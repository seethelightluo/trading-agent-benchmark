import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:get_stock_daily_data(a,5000).set_index('date').close for a in U}); R=P.pct_change(); m=R.mean(axis=1)
rows=[]
for t in P.index:
 h=R.loc[:t].tail(80).dropna(how='all'); down=h[m.loc[h.index]<0]
 if len(down)<10: continue
 # defensive payoff: positive return on broad down days relative to asset downside beta
 for a in U:
  q=down[a].dropna(); allr=h[a].dropna()
  if len(q)<8: continue
  beta=q.cov(m.loc[q.index])/m.loc[q.index].var() if m.loc[q.index].var()>0 else 0
  sig=q.mean() - beta*m.loc[q.index].mean() # residual downside payoff
  for n in [5,10,20]:
   f=P.loc[P.index>t].iloc[:n]
   if len(f)==n: rows.append((t,a,sig,n,f[a].iloc[-1]/f[a].iloc[0]-1))
out=pd.DataFrame(rows,columns=['date','asset','factor','h','fwd'])
for n,g in out.groupby('h'):
 z=pd.Series({d:q.factor.corr(q.fwd) for d,q in g.groupby('date') if len(q)>=8}).dropna(); print(n,len(z),g.asset.nunique(),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
x=out[out.h==20]; x[['date','asset','factor']].to_csv('scripts/miner_3_20300110_downside_payoff_signal.csv',index=False)
print('dates',x.date.nunique(),'coverage',x.asset.nunique()/15)
for lab,lo in [('2020-25','2020-01-01'),('2026-28','2026-01-01'),('2029','2029-01-01')]:
 z=pd.Series({d:q.factor.corr(q.fwd) for d,q in x[x.date>=lo].groupby('date') if len(q)>=8 and (lab!='2020-25' or d<=pd.Timestamp('2025-12-31')) and (lab!='2026-28' or d<=pd.Timestamp('2028-12-31'))}).dropna();print(lab,len(z),z.mean(),z.mean()/z.std(ddof=1))
