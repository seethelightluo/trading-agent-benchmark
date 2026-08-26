import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date')['close'].replace(0,np.nan)
        D[s]=x
p=pd.DataFrame(D).sort_index().ffill(limit=3)
# Candidate: 10d momentum relative to cross-sectional median, scaled by trailing vol; smooth 5-day average
r=p.pct_change()
raw=r.rolling(10).sum()/r.rolling(20).std().replace(0,np.nan)
# daily cross-sectional demean to isolate relative signal
f=raw.sub(raw.median(axis=1),axis=0)
# smooth lagged signal (all values at t predict t+1..t+H)
f=f.rolling(5,min_periods=3).mean()
rows=[]
for h in [1,5,10,20]:
  vals=[]
  for i in range(20,len(p)-h):
    a=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1
    z=pd.concat([a.rename('f'),y.rename('y')],axis=1).dropna()
    if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append((p.index[i],z.f.corr(z.y),len(z)))
  q=pd.DataFrame(vals,columns=['d','ic','n']).set_index('d')
  print('H',h,'dates',len(q),'meanIC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit', (q.ic>0).mean(),'avgN',q.n.mean())
# coverage and turnover on valid ranks
valid=f.notna().sum(axis=1); cov=f.notna().sum().sum()/f.size
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).mean()
print('range',p.index.min(),p.index.max(),'assets',len(D),'coverage',cov,'avg valid',valid.mean(),'turnover',turnover)
# regime means H10
h=10; vals=[]
for i in range(20,len(p)-h):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: vals.append((p.index[i],z.f.corr(z.y)))
q=pd.Series(dict(vals));
for a,b in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2036')]: print(a,q.loc[a:b].mean(),q.loc[a:b].count())
# artifact H10 signals
out=f.copy(); out.to_csv('scripts/miner_1_20350430_smooth_relative_momentum_signal.csv')
