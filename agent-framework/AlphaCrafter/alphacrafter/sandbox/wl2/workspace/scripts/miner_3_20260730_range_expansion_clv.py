import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=pd.Index(sorted(set.intersection(*[set(x.index) for x in D.values()])))
# asynchronous common calendar is conservative; each signal uses completed bar and is lagged.
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U}); O=pd.DataFrame({s:D[s].open.reindex(dates) for s in U})
rng=(H-L).replace(0,np.nan); clv=-(2*(P-L)/rng-1)
atr=rng.rolling(20,min_periods=12).mean(); rel=(rng/atr).clip(upper=4)
# Candidate: candle pressure weighted by unusual range, then lightly smoothed; lag avoids lookahead.
F=(clv*rel).rolling(2,min_periods=2).mean().shift(1)
Y={h:P.shift(-h).div(P).sub(1) for h in [1,5,10]}
print('range_expansion_clv universe',len(U),'dates',len(dates))
for h,y in Y.items():
 q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.array(q); print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
 if h==1:
  for k in [252,504]:
   x=q[-k:];print('recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
