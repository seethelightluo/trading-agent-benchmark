import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-24'); P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date)
 P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); m=R['US10Y']
z=(m-m.rolling(60,min_periods=40).mean())/m.rolling(60,min_periods=40).std()
vol=R.rolling(20,min_periods=15).std()
# Broader directional rate-regime reversal: lagged cross-asset reversal, active at moderate+ US10Y standardized moves.
sig=(-R.shift(1)/vol.shift(1)).mul((z.shift(1).abs()>0.75).astype(float),axis=0)

def calc(h, dates=None):
 fw=R.rolling(h).sum().shift(-h+1); q=[]; ns=[]
 for d in sig.index:
  if dates is not None and not dates[0] <= d <= dates[1]: continue
  f=sig.loc[d]; y=fw.loc[d]; ok=f.notna()&y.notna()
  if ok.sum()>=8 and f[ok].nunique()>1:
   q.append(spearmanr(f[ok],y[ok]).statistic); ns.append(ok.sum())
 q=pd.Series(q).dropna(); return len(q), np.mean(ns) if ns else np.nan, q.mean(), q.mean()/q.std(ddof=1), (q>0).mean()
print('cut',cut.date(),'assets',len(A),'raw_dates',len(sig))
for h in [1,5,10]: print('h',h,'n avg_n IC ICIR hit',calc(h))
for label,lo,hi in [('2020-22', '2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-03-24')]:
 print(label,calc(1,(pd.Timestamp(lo),pd.Timestamp(hi))))
print('coverage',sig.notna().sum(axis=1).mean()/15,'active_dates',(sig.abs().sum(axis=1)>0).sum(),'rank_turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
