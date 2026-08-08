import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Short-term reversal normalized by trailing realized risk. Lagged one day.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(-r.rolling(5,min_periods=5).sum()/vol).shift(1)
print('raw dates',len(p),'assets',len(assets),'cells',int(f.notna().sum().sum()),'coverage',f.notna().sum().sum()/f.size,'mean_valid',f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic
   if np.isfinite(z): vals.append(z); ns.append(ok.sum())
 s=pd.Series(vals); print('h=%d dates=%d meanN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
q=f.rank(axis=1,pct=True); print('turnover10',((q-q.shift(10)).abs().mean(axis=1)).mean())
for start,end in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-12')]:
 y=p.shift(-10)/p-1; vals=[]
 for dt in f.loc[start:end].index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic
   if np.isfinite(z): vals.append(z)
 s=pd.Series(vals); print('regime',start,end,'dates',len(s),'IC %.6f ICIR %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
