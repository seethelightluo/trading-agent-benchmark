import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s, days=2400)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close
P=pd.DataFrame(px).sort_index().ffill()
# use lagged factor at t based on data through t; forward return starts t+1
r=P.pct_change()
ret20=P/P.shift(20)-1
vol20=r.rolling(20).std()*np.sqrt(252)
f=ret20/vol20
# cross-sectional demean to isolate leadership
f=f.sub(f.median(axis=1),axis=0)
fr=P.shift(-10)/P-1
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
q=q.replace([np.inf,-np.inf],np.nan).dropna()
def stat(x):
 return len(x),x.n.mean(),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1), (x.ic>0).mean()
print('dates',P.index.min(),P.index.max(),'assets',len(P.columns))
print('all',stat(q),'coverage',q.n.sum()/(len(q)*15))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2028-09-01','2029-03-12')]:
 x=q.loc[a:b]
 if len(x): print(a,b,stat(x))
# turnover rank changes
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean())
q.to_csv('scripts/miner_1_20290312_peer_leadership_ic.csv')
