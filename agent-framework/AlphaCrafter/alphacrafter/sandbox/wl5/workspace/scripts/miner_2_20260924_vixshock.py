import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-09-24')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index() for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index()
dates=pd.DatetimeIndex(sorted(set().union(*[set(x.index) for x in D.values()])))
P=pd.DataFrame({s:x.close for s,x in D.items()}).reindex(dates); R=P.pct_change()
vr=v.close.pct_change().reindex(dates)
for w in [1,3,5]:
 shock=vr.rolling(5,min_periods=5).mean() # lagged macro stress, available at date
 for mode in ['all','stress']:
  f=-R.rolling(w,min_periods=w).mean() * (1+shock.clip(lower=0).to_numpy()[:,None] if mode=='stress' else (1+shock.to_numpy()[:,None]))
  ics=[]; ns=[]; ds=[]
  for i in range(len(dates)-1):
   q=pd.concat([f.iloc[i].rename('f'),R.iloc[i+1].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
    z=spearmanr(q.f,q.y).statistic
    if np.isfinite(z):ics.append(z);ns.append(len(q));ds.append(dates[i])
  x=np.array(ics); ir=x.mean()/x.std(ddof=1)
  print('w',w,mode,'dates',len(x),'N',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),5),'ICIR',round(ir,5),'hit',round(np.mean(x>0),4))
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=x[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)]; print(' regime',lo,hi,'IC',round(z.mean(),5),'n',len(z))
  yy=P.pct_change(5).shift(-5); a=[]
  for i in range(len(dates)-5):
   q=pd.concat([f.iloc[i],yy.iloc[i].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:
    z=spearmanr(q.iloc[:,0],q.y).statistic
    if np.isfinite(z):a.append(z)
  a=np.array(a);print(' h5',round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),len(a))
# save best candidate signal artifact
f=-R.rolling(3,min_periods=3).mean()*(1+shock.clip(lower=0))
f.iloc[-1].to_csv('scripts/miner_2_20260924_vixshock_signal.csv')
