import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s, days=6000)
 if d is None or len(d)<300: d=get_index_daily_data(s,days=6000)
 return d[['date','close']].drop_duplicates('date').set_index('date').close
P=pd.concat({s:get(s) for s in U},axis=1).sort_index().ffill()
r=np.log(P).diff()
# 20d residual return, conditioned on market breadth/trend: trend-follow when breadth is strong
ret20=P/P.shift(20)-1
res=ret20.sub(ret20.mean(axis=1),axis=0)
breadth=(r.rolling(20).mean()>0).mean(axis=1)
# positive breadth >= 60% selects continuation; otherwise contrarian (adaptive regime)
sgn=np.where(breadth>=.6,1.,-1.)
vol=r.rolling(20).std()*np.sqrt(252)
sig=res.mul(pd.Series(sgn,index=res.index),axis=0).div(vol.clip(lower=.04)).shift(1)
# winsorize cross section
sig=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1).replace(0,np.nan),axis=0).clip(-5,5)
rows=[]
for h in [5,10,20,40,60]:
 ic=[]; ns=[]
 fwd=P.shift(-h)/P-1
 for dt in sig.index:
  x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(ic); a=a[np.isfinite(a)]
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
sig.to_csv('scripts/miner_1_20340202_breadth_conditioned_residual_momentum_signal.csv', index_label='date')
# turnover and coverage
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('coverage',sig.notna().sum().sum()/(sig.size),'turnover',turn,'start',sig.index.min(),'end',sig.index.max())
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2034-01-31')]:
 fwd=P.shift(-10)/P-1; v=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 v=np.array(v);print('REG',a,b,len(v),round(np.nanmean(v),6),round(np.nanmean(v)/np.nanstd(v),6))
