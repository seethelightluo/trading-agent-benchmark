import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
V={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['volume'] for a in assets}
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).reindex(p.index); r=p.pct_change()
# Volume-confirmed medium-term momentum: return persistence is strengthened by rising activity.
# Lagged to ensure only completed sessions are used.
vr=np.log1p(v).rolling(20,min_periods=12).mean()-np.log1p(v).rolling(60,min_periods=36).mean()
rs=r.rolling(20,min_periods=15).sum()
f=(rs*vr).shift(1)
print('raw dates',len(p),'assets',len(assets),'cells',int(f.notna().sum().sum()),'coverage',f.notna().sum().sum()/f.size,'mean_valid',f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic); ns.append(ok.sum())
 s=pd.Series(vals); print('h=%d dates=%d meanN=%.2f IC=%.8f ICIR=%.8f hit=%.4f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 q=f.rank(axis=1,pct=True); print('turnover10',((q-q.shift(10)).abs().mean(axis=1)).mean())
for start,end in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31')]:
 y=p.shift(-10)/p-1; vals=[]
 for dt in f.loc[start:end].index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic)
 s=pd.Series(vals); print('regime',start,end,'dates',len(s),'IC %.8f ICIR %.8f'%(s.mean(),s.mean()/s.std(ddof=1)))
