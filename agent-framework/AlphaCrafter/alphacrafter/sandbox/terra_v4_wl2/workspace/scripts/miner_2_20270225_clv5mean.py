import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  x=d[['date','high','low','close']].copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date')
  D[s]=x
idx=sorted(set().union(*[set(x.index) for x in D.values()]))
# CLV is close position in completed day's range; signal lagged one day
parts={}
for s,x in D.items():
 rng=(x.high-x.low).replace(0,np.nan)
 parts[s]=((2*x.close-x.high-x.low)/rng)
clv=pd.DataFrame(parts).reindex(idx).sort_index()
f=clv.rolling(5,min_periods=4).mean().shift(1)
# use close returns and forward horizons
px=pd.DataFrame({s:x.close for s,x in D.items()}).reindex(idx).sort_index(); r=np.log(px).diff()
for h in [1,5,10]:
 y=np.log(px.shift(-h)/px)
 vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a));dates.append(dt)
 z=pd.Series(vals,index=dates); print('h',h,'dates',len(z),'avg_n',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
# rank turnover and artifact
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_clv5mean.csv',index=False);print('artifact',len(out),out.date.max())
