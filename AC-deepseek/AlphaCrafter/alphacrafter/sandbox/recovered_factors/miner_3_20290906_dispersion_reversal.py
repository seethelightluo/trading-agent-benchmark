import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 px[a]=d['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Factor: contrarian short-term return, scaled by contemporaneous cross-sectional dispersion,
# with 60d smoothed dispersion state. All rolling values are shifted before forward return.
csdisp=r.rolling(20,min_periods=15).std(axis=1)
disp_state=csdisp.rolling(60,min_periods=40).mean()
short=r.rolling(5,min_periods=4).sum()
f=(-short.sub(short.mean(axis=1),axis=0)).div(r.rolling(20,min_periods=15).std())
f=f.mul(disp_state,axis=0)
# Explicitly lag signal one completed day
f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1
 vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 s=pd.Series(vals)
 rows.append((h,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('candidate=dispersion_weighted_relative_reversal; dates=%d assets=%d'% (len(p),len(assets)))
for z in rows: print('h=%d dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%z)
# 10d rank turnover proxy
rr=f.rank(axis=1,pct=True); turn=(rr-rr.shift(10)).abs().mean(axis=1).mean()
print('coverage=%.6f turnover10=%.6f mean_valid=%.3f'%(f.notna().sum().sum()/f.size,turn,f.notna().sum(axis=1).mean()))
# recent regimes
for start,end in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-09-06')]:
 fr=p.shift(-10)/p-1; vals=[]
 for dt in f.loc[start:end].index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic)
 s=pd.Series(vals); print('regime %s/%s n=%d IC=%.6f ICIR=%.6f'%(start,end,len(s),s.mean(),s.mean()/s.std(ddof=1)))
