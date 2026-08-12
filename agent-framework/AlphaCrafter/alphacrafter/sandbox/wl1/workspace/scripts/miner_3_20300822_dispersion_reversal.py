import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  q=d.copy();q.date=pd.to_datetime(q.date);F[s]=q.set_index('date').close.astype(float)
p=pd.concat(F,axis=1).sort_index().ffill(); r=p.pct_change()
# Contrarian 5-day return, activated when cross-asset dispersion is elevated; lag one day.
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
threshold=disp.rolling(120,min_periods=60).median()
active=(disp>threshold).astype(float)
s=(-r.rolling(5).sum()*active.values[:,None]).shift(1)
print('assets',len(F),'dates',len(p),'disp active',active.mean())
for h in [1,5,10,20]:
 f=p.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(a).dropna();print(f'h{h} dates={len(a)} IC={a.mean():.6f} ICIR={a.mean()/a.std():.6f} hit={(a>0).mean():.4f}')
print('avgN',np.mean(ns),'coverage',np.mean(ns)/15,'turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean())
for y in sorted(set(s.index.year)):
 a=[]
 for dt in s.index[s.index.year==y]:
  z=pd.concat([s.loc[dt],p.pct_change().shift(-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if len(a)>20:
  a=pd.Series(a);print(y,len(a),a.mean(),a.mean()/a.std(),(a>0).mean())
s.to_csv('scripts/miner_3_20300822_dispersion_reversal_signal.csv')
