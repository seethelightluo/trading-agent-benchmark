import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={s:get_stock_daily_data(s,days=6000).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(frames.values(),axis=1).sort_index().ffill(); m=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
r=p.pct_change(); mr=m.pct_change(); vm=mr.rolling(60,min_periods=40).var(); beta=pd.DataFrame({s:r[s].rolling(60,min_periods=40).cov(mr)/vm for s in U})
res=p.pct_change(20)-beta.mul(m.pct_change(20),axis=0); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20); f=(-res/vol).replace([np.inf,-np.inf],np.nan); fr=p.shift(-10).div(p)-1
rows=[];dates=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
ic=pd.Series(rows,index=dates).dropna(); print('factor=60d DXY-beta residual 20d reversal/vol; dates',len(ic),'meanN',np.mean(ns),'coverage',len(ic)/len(f)); print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2023'),('2024','2025'),('2026','2028'),('2029','2031'),('2032','2035')]:
 q=ic.loc[a:b]; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.loc['2020':].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20350830_dxy_beta_residual_reversal_signal.csv',index=False); print('artifact rows',len(out))
