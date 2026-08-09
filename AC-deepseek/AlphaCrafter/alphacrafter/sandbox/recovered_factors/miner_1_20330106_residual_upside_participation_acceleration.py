"""One candidate only: residual upside participation acceleration.
Higher score means idiosyncratic positive-return frequency over the latest 20
completed sessions exceeds its 60-session baseline.  Tests whether improving
asset-specific participation forecasts future cross-asset relative return.
Data cutoff is the last completed bar before the 2033-01-06 decision date.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
START=pd.Timestamp('2026-07-16'); END=pd.Timestamp('2033-01-05')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in AS}
idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.loc[START:END].index) for x in D.values()])))
c=pd.DataFrame({a:D[a].reindex(idx)['close'] for a in AS}); r=c.pct_change(); med=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0); resid=r-beta.mul(med,axis=0)
s=resid.gt(0).rolling(20,min_periods=15).mean()-resid.gt(0).rolling(60,min_periods=45).mean()
print('FACTOR residual_upside_participation_acceleration_20_60obs')
print('cutoff',END.date(),'assets',len(AS),'calendar_dates',len(idx),'coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
out={}
for h in [1,5,10,20]:
 f=c.shift(-h).div(c).sub(1); v=[]; n=[]; dates=[]
 for t in idx:
  x=pd.concat([s.loc[t],f.loc[t]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   z=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
   if np.isfinite(z):v.append(z);n.append(len(x));dates.append(t)
 v=np.array(v); n=np.array(n); dates=pd.DatetimeIndex(dates); out[h]=(v,dates)
 print('H',h,'dates',len(v),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),6),'mean_n',round(n.mean(),3),'min_n',n.min(),'PASS',abs(v.mean())>=.007 and abs(v.mean()/v.std(ddof=1)>=.084))
v,dates=out[20]
for name,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_2031','2030-01-01','2031-12-31'),('recent_12m','2032-01-06',END)]:
 x=v[(dates>=lo)&(dates<=hi)];print('REGIME',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=s.rank(axis=1,pct=True); print('turnover',round((rk-rk.shift()).abs().stack().mean(),6),'comparisons',len(idx)-1,'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))
print('LIBRARY_AUDIT', 'NOT_PRODUCED: complete same-date signals for all 30 admitted factors are not reconstructible from definitions alone; candidate cannot be admitted without required evidence.')
