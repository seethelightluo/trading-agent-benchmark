import pandas as pd, numpy as np, glob, os
from pathlib import Path
from scipy.stats import pearsonr

CUT=pd.Timestamp('2031-10-16')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
 p=Path('../persistent/stock_data')/(sym+'.csv')
 d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index()
 return d[d.index<=CUT]
px=pd.concat({s:load(s) for s in assets},axis=1)
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
macro=macro[macro.index<=CUT].reindex(px.index).ffill()
r=np.log(px).diff(); dr=np.log(macro).diff()
# beta residual 20d return over rolling 60d covariance, strictly lagged in signal
cov=r.rolling(60,min_periods=40).cov(dr)
var=dr.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0)
r20=np.log(px/px.shift(20)); d20=np.log(macro/macro.shift(20))
res=r20-beta.shift(1).mul(d20,axis=0)
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
sig=(res/vol).shift(1)
fwd=np.log(px.shift(-10)/px)
ics=[]; turnovers=[]; counts=[]
prev=None
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(x[ok].corr(y[ok])); counts.append(ok.sum())
  ranks=x.rank();
  if prev is not None: turnovers.append((ranks[ok]-prev[ok]).abs().mean()/len(assets))
  prev=ranks
s=pd.Series(ics).dropna()
print('factor=20D DXY-beta residual momentum / 30D vol, lag1')
print('dates',len(s),'avg_names',np.mean(counts),'coverage',np.mean(counts)/15,'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turnover',np.mean(turnovers))
for label,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-12-31'),('2029-31','2029-01-01','2031-10-16'),('recent120',None,None)]:
 z=s if label=='recent120' else s.iloc[0:0] # IC index not retained below
# recompute indexed IC for regime output
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok])))
 si=pd.Series(dict(rows))
for label,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-12-31'),('2029-31','2029-01-01','2031-10-16'),('recent120',None,None)]:
 z=si.tail(120) if label=='recent120' else si.loc[a:b]
 print(label,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if z.std()>0 else None)
# artifact, aligned date/symbol signal
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311016_dxy_residual_momentum_signal.csv',index=False)
print('artifact rows',len(out))
