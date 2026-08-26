import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: D[s]=d[['date','close','high','low']].drop_duplicates('date').set_index('date')
P=pd.concat({s:x.close for s,x in D.items()},axis=1).sort_index().ffill(); R=np.log(P).diff()
H=pd.concat({s:x.high for s,x in D.items()},axis=1).reindex(P.index).ffill(); L=pd.concat({s:x.low for s,x in D.items()},axis=1).reindex(P.index).ffill()
rng=(H-L)/P.replace(0,np.nan)
# Smooth 3-session shock and range expansion, then reverse; lag one completed session.
vol=R.rolling(20,min_periods=12).std()
shock=(R/vol).rolling(3,min_periods=3).mean()
range_ratio=(rng/(rng.rolling(20,min_periods=12).median()+1e-12)).rolling(3,min_periods=3).mean()
raw=(-shock*range_ratio.clip(0,5)).clip(-10,10)
f=raw.sub(raw.median(axis=1),axis=0).shift(1)
fr={h:np.log(P.shift(-h)/P) for h in [1,3,5,10,20]}
all_ic={}
for h in fr:
 q=[];ns=[];ds=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 x=pd.Series(q,index=ds).dropna();all_ic[h]=x
 print(f'H{h} dates={len(x)} avg_n={np.mean(ns):.2f} IC={x.mean():.8f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.8f} hit={(x>0).mean():.4f}')
x=all_ic[1]; n=len(x)
print('regimes',*[f'{x.iloc[a:b].mean():.8f}' for a,b in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
print('recent252',x.tail(252).mean(),'recent756',x.tail(756).mean())
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(P),'instruments',len(D))
x.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20310210_smoothed_range_reversal_ic.csv',index=False)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310210_smoothed_range_reversal_signal.csv',index=False)
