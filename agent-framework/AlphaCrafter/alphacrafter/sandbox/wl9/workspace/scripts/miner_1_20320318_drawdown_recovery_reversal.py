import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4400)
 if d is not None and len(d)>300:
  C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); r120=p.pct_change(120)
# Drawdown-recovery reversal: favor assets with deep lagged drawdowns versus the cross-sectional median,
# scaled by their recovery efficiency (recent 20d return relative to 60d downside volatility).
med=r120.median(axis=1); rel=r120.sub(med,axis=0)
rollmax=p.rolling(120,min_periods=60).max(); dd=p/rollmax-1
recent=p.pct_change(20)
down=r.where(r<0,0.0); dv=down.rolling(60,min_periods=30).std()*np.sqrt(252)
recovery=(recent/(dv+1e-12)).clip(-3,3)
f=(-rel)*(1+0.25*recovery.clip(lower=0))
f=f.shift(1)
print('UNIVERSE',len(C),'DATES',len(p),'assets',sorted(C))
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-03-17')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320318_drawdown_recovery_reversal_signal.csv',index=False)
