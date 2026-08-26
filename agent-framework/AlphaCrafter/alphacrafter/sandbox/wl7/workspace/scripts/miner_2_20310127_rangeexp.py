import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  D[s]=d[['date','open','close','high','low']].drop_duplicates('date').set_index('date')
P=pd.concat({s:x.close for s,x in D.items()},axis=1).sort_index().ffill(); R=np.log(P).diff()
O=pd.concat({s:x.open for s,x in D.items()},axis=1).reindex(P.index).ffill(); H=pd.concat({s:x.high for s,x in D.items()},axis=1).reindex(P.index).ffill(); L=pd.concat({s:x.low for s,x in D.items()},axis=1).reindex(P.index).ffill()
# Range-expansion reversal: recent close-to-close shock, amplified when its true daily range is unusually large.
# All inputs are lagged one completed session before scoring.
range_pct=(H-L)/P.replace(0,np.nan)
range_z=range_pct/(range_pct.rolling(20,min_periods=12).median()+1e-12)
shock=R/(R.rolling(20,min_periods=12).std()+1e-12)
raw=-shock*range_z.clip(0,5)
f=raw.sub(raw.median(axis=1),axis=0).shift(1)
for h in [1,3,5,10,20]:
 q=[]; ns=[]; ds=[]; fr=np.log(P.shift(-h)/P)
 for dt in P.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 x=pd.Series(q,index=ds).dropna();print(f'H{h} dates={len(x)} avg_n={np.mean(ns):.2f} IC={x.mean():.8f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.8f} hit={(x>0).mean():.4f}')
 if h==1:x.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20310127_rangeexp_ic.csv',index=False)
q=[];ds=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt)
x=pd.Series(q,index=ds).dropna();n=len(x)
print('regimes',*[f'{x.iloc[a:b].mean():.8f}' for a,b in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
print('recent252',x.tail(252).mean(),'recent756',x.tail(756).mean())
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(P),'instruments',len(D))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310127_rangeexp_signal.csv',index=False)
