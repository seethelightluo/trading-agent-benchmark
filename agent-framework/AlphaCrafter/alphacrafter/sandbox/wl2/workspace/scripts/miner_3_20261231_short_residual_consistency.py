import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-12-30')); dates=D['SPX'].index[(D['SPX'].index>='2020-03-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change(); M=R.mean(axis=1)
b=R.rolling(40,min_periods=25).cov(M).div(M.rolling(40,min_periods=25).var(),axis=0); res=R-b.mul(M,axis=0)
# Short-horizon residual trend, gated by directional consistency and scaled by idiosyncratic volatility.
tr=res.rolling(10,min_periods=8).sum(); cons=(res>0).rolling(10,min_periods=8).mean(); vol=res.rolling(20,min_periods=12).std()
F=(tr*(2*cons-1)/vol).shift(1)
for h in [1,3,5]:
 y=C.shift(-h).div(C)-1; a=[]; ds=[]; ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
  q=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
