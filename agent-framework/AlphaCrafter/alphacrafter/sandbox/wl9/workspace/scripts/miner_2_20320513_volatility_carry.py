import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,4500)
 if d is not None and len(d)>300:C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); vol=r.rolling(60,min_periods=40).std().replace(0,np.nan); cross=vol.median(axis=1).replace(0,np.nan)
f=pd.DataFrame(np.log(cross.values[:,None])-np.log(vol.values),index=p.index,columns=p.columns).clip(-2,2).shift(1)
print('data',p.index.min(),p.index.max(),'rows',len(p),'instruments',len(C))
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; q=[];n=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z));ds.append(dt)
 a=pd.Series(q,index=pd.to_datetime(ds)).dropna();print(h,'dates',len(a),'avgN %.2f'%np.mean(n),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
 if h==20:
  for aa,bb in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-05-12')]:
   z=a.loc[aa:bb];print('REGIME',aa[:4],len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()))
print('coverage %.6f turnover %.6f'%((f.notna().mean().mean()),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320513_volatility_carry_signal.csv',index=False)
