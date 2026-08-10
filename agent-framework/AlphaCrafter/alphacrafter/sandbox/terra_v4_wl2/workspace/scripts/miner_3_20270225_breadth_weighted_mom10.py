import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut='2027-02-25'
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 xs[s]=d
p=pd.DataFrame(xs).loc[:cut]
r=p.pct_change()
# breadth-weighted 10d momentum: own trailing return times contemporaneous breadth, with sign-preserving strength
mom=p.pct_change(10)
breadth=((r>0).rolling(10,min_periods=8).mean().mean(axis=1)-.5)*2
sig=mom.mul(breadth,axis=0)
# forward 1d, 5d, 10d
out=[]
for h in [1,5,10]:
 fwd=p.shift(-h).div(p)-1
 ics=[]; n=[]; dates=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=fwd.loc[dt]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(rho): ics.append(rho); n.append(len(z)); dates.append(dt)
 q=np.array(ics); print(h,'dates',len(q),'avg_n',round(np.mean(n),2),'IC',round(np.mean(q),5),'ICIR',round(np.mean(q)/(np.std(q,ddof=1)+1e-12),5),'hit',round(np.mean(q>0),4))
# turnover and coverage
valid=sig.notna().sum(axis=1)/15
turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1)
print('coverage_date_mean',valid.mean(),'matrix_coverage',sig.notna().mean().mean(),'turnover',turn.mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 z=[]
 for dt in sig.loc[a:b].index:
  x=pd.concat([sig.loc[dt],p.pct_change().shift(-1).loc[dt]],axis=1).dropna()
  if len(x)>=8:
   rho=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
   if np.isfinite(rho): z.append(rho)
 print(a,b,len(z),round(np.mean(z),5) if z else None)
# save artifact
art=pd.DataFrame(sig).stack().rename('signal').reset_index();art.columns=['date','symbol','signal'];art.to_csv('../persistent/factor_signals_miner_3_20270225_breadth_weighted_mom10.csv',index=False)
