import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-04-07'); P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date)
 P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); med=R.median(axis=1)
# Cross-sectional short-horizon reversal, only when dispersion is elevated; all inputs lagged.
csdisp=R.sub(med,axis=0).pow(2).mean(axis=1).pow(.5)
gate=(csdisp>csdisp.rolling(60,min_periods=40).quantile(.65)).astype(float)
sig=-(R.rolling(5,min_periods=5).sum().shift(1).sub(R.rolling(5,min_periods=5).sum().shift(1).median(axis=1),axis=0)).mul(gate.shift(1),axis=0)
fw={h:R.rolling(h).sum().shift(-h+1) for h in [1,5,10]}
def calc(h, dates=None):
 q=[]; ns=[]
 for d in sig.index:
  if dates is not None and not dates[0]<=d<=dates[1]: continue
  f=sig.loc[d]; y=fw[h].loc[d]; ok=f.notna()&y.notna()
  if ok.sum()>=8 and f[ok].nunique()>1:
   q.append(spearmanr(f[ok],y[ok]).statistic); ns.append(ok.sum())
 q=pd.Series(q).dropna(); return len(q),round(np.mean(ns),2),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),round((q>0).mean(),4)
print('cut',cut.date(),'assets',len(A),'raw_dates',len(sig))
for h in [1,5,10]: print('h',h,'n avg_n IC ICIR hit',calc(h))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-04-07')]: print(label,calc(1,(pd.Timestamp(lo),pd.Timestamp(hi))))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'active_dates',int((sig.abs().sum(axis=1)>0).sum()),'rank_turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
