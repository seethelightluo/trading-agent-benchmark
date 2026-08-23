import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);p[s]=d.sort_values('date').set_index('date').close.astype(float)
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
# residual cumulative move versus contemporaneous cross-asset mean, reversed and risk scaled
res=(r.sub(m,axis=0)).rolling(10,min_periods=8).sum()
vol=r.rolling(30,min_periods=20).std()*np.sqrt(252)
f=-res/(vol+0.02)
ics=[]; ns=[]; ts=[]; dates=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  q=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(q):ics.append(q);ns.append(ok.sum());dates.append(p.index[i]);ts.append((x[ok].rank(pct=True)-f.iloc[i-1][ok].rank(pct=True)).abs().mean() if i else np.nan)
a=np.array(ics);print(json.dumps({'factor':'residual_reversal_10d','dates':len(a),'avg_instruments':np.mean(ns),'coverage':np.mean(ns)/15,'IC':np.mean(a),'ICIR':np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),'hit_ratio':np.mean(a>0),'turnover_proxy':np.nanmean(ts)},indent=2))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 z=[v for d,v in zip(dates,a) if str(d)[:4]>=lo and str(d)[:10]<=hi];print(lo, np.mean(z) if z else None,len(z))
