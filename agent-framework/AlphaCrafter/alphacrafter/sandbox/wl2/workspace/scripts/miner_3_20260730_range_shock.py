import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
# One idea: lagged abnormal intraday range (shock/reversal), normalized by 20d median range.
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); O=pd.DataFrame({s:D[s].open.reindex(dates) for s in U})
H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U})
rng=(H-L)/C.shift(0)
base=rng.rolling(20,min_periods=12).median()
# negative abnormal range: unusually wide sessions expected to mean revert cross-sectionally
F=-(rng/base-1).shift(1)
# alternative robust log shock, same economic idea
F2=-np.log((rng/base).clip(lower=1e-6)).shift(1)
Y=pd.DataFrame({s:D[s].close.shift(-1).div(D[s].close).sub(1).reindex(dates) for s in U})
for label,fac in [('range_shock',F),('log_range_shock',F2)]:
 q=[];ns=[];ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':fac.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q); print(label,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(fac.notna().sum().sum()/fac.size,4))
 for yr in range(2020,2027):
  x=np.array([q[i] for i,d in enumerate(ds) if d.year==yr]);
  if len(x): print(' regime',yr,len(x),round(x.mean(),6))
 print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
 print('decay',end=' ')
 for h in [5,10]:
  yy=pd.DataFrame({s:D[s].close.shift(-h).div(D[s].close).sub(1).reindex(dates) for s in U}); qq=[]
  for dt in dates:
   z=pd.DataFrame({'f':fac.loc[dt],'y':yy.loc[dt]}).dropna()
   if len(z)>=8: qq.append(spearmanr(z.f,z.y).statistic)
  print(h,round(np.mean(qq),6),end='; ')
 print()
print('paircorr',F.stack().rank().corr(F2.stack().rank()))
