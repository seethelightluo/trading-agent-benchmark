import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5200)
 if d is not None and len(d)>200:
  x=d[['date','close']].drop_duplicates('date').set_index('date').sort_index()
  P[s]=x.close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); lr=np.log(p).diff(); ret30=np.log(p/p.shift(30)); vol20=lr.rolling(20,min_periods=15).std()
# Trend quality: residual 30-session displacement, volatility-scaled and
# weighted by path efficiency (net displacement / absolute daily path).
res=ret30.sub(ret30.median(axis=1),axis=0)
path=lr.abs().rolling(30,min_periods=20).sum()
eff=(ret30.abs()/path).clip(0,1)
sig=(res/vol20*eff).shift(1)
fwd=np.log(p.shift(-10)/p)
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor efficiency_weighted_residual_trend30 dates',len(r),'avgN',r.n.mean(),'coverage',r.n.mean()/15,'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'turn',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 q=r.tail(n); print('recent',n,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
for h in [5,10,20]:
 yy=np.log(p.shift(-h)/p); rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(rr).dropna(); print('horizon',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'obs',len(q))
out=sig.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_2_20340203_efficiency_weighted_residual_trend30_signal.csv')
r.to_csv('scripts/miner_2_20340203_efficiency_weighted_residual_trend30_ic.csv')
