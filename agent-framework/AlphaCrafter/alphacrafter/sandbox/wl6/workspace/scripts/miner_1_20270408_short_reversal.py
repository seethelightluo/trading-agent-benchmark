import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-04-07');P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); x=R.rolling(5,min_periods=5).sum().shift(1); sig=-x.sub(x.median(axis=1),axis=0); fw={h:R.rolling(h).sum().shift(-h+1) for h in [1,5,10]}
def calc(h,ab=None):
 q=[];ns=[]
 for d in sig.index:
  if ab and not ab[0]<=d<=ab[1]:continue
  f=sig.loc[d];y=fw[h].loc[d];ok=f.notna()&y.notna()
  if ok.sum()>=8 and f[ok].nunique()>1:q.append(spearmanr(f[ok],y[ok]).statistic);ns.append(ok.sum())
 q=pd.Series(q).dropna();return len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('cut',cut.date(),'assets',len(A))
for h in [1,5,10]:print(h,calc(h))
for z,l,u in [('20-22','2020-01-01','2022-12-31'),('23-24','2023-01-01','2024-12-31'),('25-26','2025-01-01','2026-12-31'),('27','2027-01-01','2027-04-07')]:print(z,calc(5,(pd.Timestamp(l),pd.Timestamp(u))))
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())