import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
d={s:ld(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1); r=np.log(c).diff()
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std(); ratio=v20.mean(axis=1)/v60.mean(axis=1)
# Cross-asset volatility transition: during elevated aggregate volatility, reverse 5d shocks,
# scaled by asset volatility; signal is lagged.
stress=ratio>ratio.rolling(252,min_periods=126).quantile(.75)
f=(-(r.rolling(5).sum())/(v20*np.sqrt(5))).shift(1)
res=[]
for dt in r.index:
 if not bool(stress.loc[dt]): continue
 v=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=v.notna()&y.notna()
 if ok.sum()>=8: res.append((dt,spearmanr(v[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('stress_dates',len(x),'assets',len(U),'coverage',x.n.mean()/15 if len(x) else 0,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-05-27')]:
 z=x.loc[a:b]; print('regime',a,len(z),z.ic.mean() if len(z) else np.nan,z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h); q=[]
 for dt in r.index:
  if not bool(stress.loc[dt]): continue
  v=f.loc[dt]; yy=y.loc[dt]; ok=v.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(v[ok],yy[ok]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
f.to_csv('scripts/miner_3_20330527_vol_transition_reversal_signal.csv')
