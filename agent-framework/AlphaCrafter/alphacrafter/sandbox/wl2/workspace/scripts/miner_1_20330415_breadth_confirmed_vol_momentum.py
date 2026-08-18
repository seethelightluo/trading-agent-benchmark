import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close.sort_index()
p=pd.concat({s:load(s) for s in U},axis=1); r=np.log(p).diff()
# Single idea: lagged volatility-normalized 10d momentum, confirmed by lagged cross-sectional breadth.
ret=r.rolling(10,min_periods=10).sum(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(10)
breadth=(ret>0).mean(axis=1).rolling(5,min_periods=5).mean()
f=(ret/vol).where(breadth.shift(1)>0.55).shift(1)
rows=[]
for d in r.index:
 v=f.loc[d]; y=r.shift(-1).loc[d]; ok=v.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,spearmanr(v[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'turnover',np.nan)
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-04-14')]:
 z=x.loc[a:b]; print('regime',a,b,'dates',len(z),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h); q=[]
 for d in r.index:
  v=f.loc[d]; yy=y.loc[d]; ok=v.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(v[ok],yy[ok]).statistic)
 print('decay',h,'dates',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1))
f.to_csv('scripts/miner_1_20330415_breadth_confirmed_vol_momentum_signal.csv')
