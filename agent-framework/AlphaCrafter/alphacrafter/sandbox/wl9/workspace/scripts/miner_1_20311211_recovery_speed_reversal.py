import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
series={}
for s in U:
 d=get_stock_daily_data(s,days=4200)
 if d is not None and len(d)>300:
  series[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(series).sort_index(); r=np.log(p).diff()
# Recovery-speed reversal: fade medium-term residual losers, weighting assets whose recent
# positive-session magnitude indicates faster recovery from prior losses. Lagged one day.
res=p.pct_change(60).sub(p.pct_change(60).mean(axis=1),axis=0)
pos=r.clip(lower=0).rolling(40,min_periods=15).mean()
neg=(-r.clip(upper=0)).rolling(40,min_periods=15).mean()
recovery=(pos/(neg+1e-8)).clip(0.25,4.0)
f=(-res*recovery).shift(1)
frs={h:p.shift(-h)/p-1 for h in [5,10,20,40,60]}; allq={}
for h,fr in frs.items():
 vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); allq[h]=q
 print('H',h,'valid_dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
q=allq[20]
coverage=[]
for d in q.index: coverage.append(len(pd.concat([f.loc[d],frs[20].loc[d]],axis=1).dropna())/len(U))
print('DETAIL dates',len(q),'universe',len(U),'coverage %.4f turnover %.6f'%(np.mean(coverage),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-12-10')]:
 z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311211_recovery_speed_reversal_signal.csv',index=False)
