import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
S={}
for s in U:
 d=get_stock_daily_data(s,days=4200)
 if d is not None and len(d)>300: S[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(S).sort_index(); r=np.log(p).diff()
# Trend continuation: intermediate (60d) residual momentum, confirmed only when it agrees
# with slower (180d) residual momentum; normalize by 60d volatility and lag one session.
cs=r.sub(r.mean(axis=1),axis=0)
m60=cs.rolling(60,min_periods=40).sum(); m180=cs.rolling(180,min_periods=100).sum()
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
agree=np.sign(m60)*np.sign(m180)
f=(m60/vol)*np.where(agree>0,1.0,0.35)
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==20:
  for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-12-24')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
  print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20311225_dual_horizon_momentum_signal.csv',index=False)
