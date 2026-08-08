import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
p=pd.DataFrame({a:d[a].close for a in assets}).sort_index()
# Persistent directional close-location pressure: average daily close location in its range,
# multiplied by recent return efficiency; lagged one completed session.
hi=pd.DataFrame({a:d[a].high for a in assets}).reindex(p.index)
lo=pd.DataFrame({a:d[a].low for a in assets}).reindex(p.index)
loc=((p-lo)/(hi-lo).replace(0,np.nan)-.5).clip(-.5,.5)
r=p.pct_change(); eff=r.rolling(10,min_periods=8).sum()/(r.abs().rolling(10,min_periods=8).sum()+1e-12)
f=(loc.rolling(10,min_periods=8).mean()*eff).shift(1)
print('raw dates',len(p),'assets',len(assets),'cells',int(f.notna().sum().sum()),'coverage',f.notna().sum().sum()/f.size,'mean_valid',f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic
   if np.isfinite(z): vals.append(z);ns.append(ok.sum())
 s=pd.Series(vals); print('h=%d dates=%d meanN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
y=p.shift(-10)/p-1
for start,end in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-26')]:
 vals=[]
 for dt in f.loc[start:end].index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic
   if np.isfinite(z): vals.append(z)
 s=pd.Series(vals); print('regime',start,end,'dates',len(s),'IC %.6f ICIR %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
