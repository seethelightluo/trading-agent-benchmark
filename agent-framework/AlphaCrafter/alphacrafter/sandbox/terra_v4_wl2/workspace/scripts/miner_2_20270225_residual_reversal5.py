import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s, days=3000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
  frames[s]=x
px=pd.DataFrame(frames).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1)
# 5-session residual reversal, beta estimated using only prior data via rolling covariance then lag signal
cov=r.rolling(60,min_periods=30).cov(bench); var=bench.rolling(60,min_periods=30).var()
beta=cov.div(var,axis=0); resid=r-beta.mul(bench,axis=0)
factor=-resid.rolling(5,min_periods=5).sum().shift(1)
fwd=r.shift(-1)
ics=[]; rows=[]
for dt in factor.index:
 a=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  ic=a.iloc[:,0].corr(a.iloc[:,1],method='spearman'); ics.append(ic); rows.append((dt,ic,len(a)))
ics=pd.Series([x[1] for x in rows],index=[x[0] for x in rows])
print('dates',len(ics),'avg_n',np.mean([x[2] for x in rows]),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'coverage',len(ics)/len(factor.index))
for n in [5,10,20]:
 ff=-resid.rolling(5,min_periods=5).sum().shift(1); yy=r.shift(-n).rolling(n).sum().shift(-(n-1))
 z=[]
 for dt in ff.index:
  a=pd.concat([ff.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 print('decay',n,np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1))
# save aligned signal
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_2_20270225_residual_reversal5.csv',index=False)
print('artifact rows',len(out),'last',out.date.max())
