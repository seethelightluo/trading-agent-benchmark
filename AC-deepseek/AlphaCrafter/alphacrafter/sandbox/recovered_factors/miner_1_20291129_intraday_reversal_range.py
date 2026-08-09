import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for a in A:d[a]=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
p=pd.DataFrame({a:d[a].close for a in A}).sort_index(); op=pd.DataFrame({a:d[a].open for a in A}).reindex(p.index); hi=pd.DataFrame({a:d[a].high for a in A}).reindex(p.index); lo=pd.DataFrame({a:d[a].low for a in A}).reindex(p.index)
# Fade the signed intraday move, normalized by true range, with a 3-day smoothing to reduce noise.
rng=(hi-lo).div(p.shift(1)).replace([np.inf,-np.inf],np.nan); signed=(p-op).div(p.shift(1)).div(rng).replace([np.inf,-np.inf],np.nan)
f=(-signed.rolling(3,min_periods=2).mean()).shift(1)
for k in [1,5,10,20]:
 y=p.shift(-k)/p-1; z=[]; ns=[]
 for t in f.index:
  ok=f.loc[t].notna()&y.loc[t].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[t,ok],y.loc[t,ok]).statistic);ns.append(ok.sum())
 s=pd.Series(z);print('h=%d dates=%d meanN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(k,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage=%.6f mean_valid=%.3f turnover10=%.6f'%(f.notna().sum().sum()/f.size,f.notna().sum(axis=1).mean(),f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean()))
for q,s,e in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-29','2028','2029-11-28')]:
 y=p.shift(-5)/p-1;z=[]
 for t in f.loc[s:e].index:
  ok=f.loc[t].notna()&y.loc[t].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[t,ok],y.loc[t,ok]).statistic)
 x=pd.Series(z);print('regime=%s n=%d IC=%.6f ICIR=%.6f'%(q,len(x),x.mean(),x.mean()/x.std(ddof=1)))
