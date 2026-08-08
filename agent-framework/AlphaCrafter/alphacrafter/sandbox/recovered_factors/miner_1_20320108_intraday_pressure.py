import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 # signed intraday close-location, avoiding zero ranges
 rng=(d.high-d.low).replace(0,np.nan)
 P[a]=((d.close-d.open)/rng).rename(a)
L=pd.DataFrame(P).sort_index()
# persistent directional intraday pressure, lagged by construction at decision time
F=L.rolling(10,min_periods=7).mean()
# cross-sectional demean is rank-equivalent; evaluate forward close returns
C=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}).sort_index()
R=C.pct_change()
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]; turnover=[]; prev=None
 for i in range(10,len(C)-h):
  z=pd.concat([F.iloc[i], C.iloc[i+h]/C.iloc[i]-1],axis=1).dropna()
  if len(z)<8: continue
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): vals.append(q);dates.append(C.index[i]);ns.append(len(z))
  r=F.iloc[i].rank(pct=True)
  if prev is not None: turnover.append(np.mean((r-prev).abs()))
  prev=r
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(turnover)))
 for lo,hi,n in [('2020','2024','20-23'),('2024','2028','24-27'),('2028','2031','28-30'),('2031','2033','31+')]:
  q=a[(np.array(dates)>=pd.Timestamp(lo+'-01-01'))&(np.array(dates)<pd.Timestamp(hi+'-01-01'))]
  if len(q): print(' ',n,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',F.notna().mean().mean(),'valid_dates',len(F))
print('DECAY: use same signal, horizons 1/5/10/20 above')
