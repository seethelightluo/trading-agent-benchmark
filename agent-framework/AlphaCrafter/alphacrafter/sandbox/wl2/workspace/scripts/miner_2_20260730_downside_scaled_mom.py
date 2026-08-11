import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
idx=pd.Index(sorted(set().union(*[set(x.index) for x in D.values()])))
P=pd.DataFrame({s:D[s].close.reindex(idx) for s in U}); R=P.pct_change()
# Candidate: medium momentum penalized only by downside risk; all signal inputs lagged one day.
down=R.where(R<0).rolling(30,min_periods=15).std()
F=P.pct_change(20).div(down).shift(1)
Y={h:P.shift(-h).div(P)-1 for h in [1,5,10]}
print('idea downside_scaled_momentum_20_30; universe',len(U),'dates',len(idx))
for h,y in Y.items():
 q=[]; ns=[]; ds=[]
 for dt in idx:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
 if h==1:
  for yr in range(2020,2027):
   x=q[[d.year==yr for d in ds]]
   print('regime',yr,len(x),round(x.mean(),6) if len(x) else None,round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
print('coverage_all',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.to_csv('scripts/miner_2_20260730_downside_scaled_mom_signal.csv')
