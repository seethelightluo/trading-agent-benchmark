import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
q={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)==0:d=get_index_daily_data(s,days=6000)
 if d is not None and len(d):q[s]=pd.Series(d.close.values,index=pd.to_datetime(d.date)).groupby(level=0).last()
p=pd.DataFrame(q).sort_index(); r=np.log(p).diff()
# continuation signal: 30d return, penalized only by downside volatility; lagged one day
ret=p.pct_change(30)
down=r.where(r<0).rolling(30,min_periods=15).std()
f=(ret/(down+1e-8)).shift(1)
for h in [5,10,20,40]:
 fr=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a))
 z=pd.Series(z).dropna();print('H',h,'obs',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
 if h==20: sel=z
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2027-12-31'),('2028','2031-12-31'),('2032','2035-03-14')]:
 z=sel.loc[a:b];print('regime',a,b,len(z),z.mean())
f.to_csv('scripts/miner_1_20350315_downside_sharpe_continuation_signal.csv',index_label='date')
