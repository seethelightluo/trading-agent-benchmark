import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ds={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
# Intraday signed efficiency: close-open relative to true range, averaged 10d; lagged. Positive means buyers controlled session.
o=pd.DataFrame({a:d.open for a,d in ds.items()}); c=pd.DataFrame({a:d.close for a,d in ds.items()}); hi=pd.DataFrame({a:d.high for a,d in ds.items()}); lo=pd.DataFrame({a:d.low for a,d in ds.items()})
r=c.pct_change(); rng=(hi-lo)/c.shift(1); eff=((c-o)/c.shift(1))/rng.replace(0,np.nan)
f=(-eff.rolling(10,min_periods=7).mean()).shift(1).clip(-5,5)
print('dates',len(c),'assets',len(A),'coverage',f.notna().sum().sum()/f.size,'mean_valid',f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 y=c.shift(-h)/c-1; vals=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic); ns.append(ok.sum())
 s=pd.Series(vals); print('h=%d dates=%d meanN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
q=f.rank(axis=1,pct=True); print('turnover10',((q-q.shift(10)).abs().mean(axis=1)).mean())
for st,en in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-10-31')]:
 y=c.shift(-10)/c-1; x=[]
 for dt in f.loc[st:en].index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:x.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic)
 s=pd.Series(x);print('regime',st,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1))
