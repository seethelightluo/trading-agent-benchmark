import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2027-04-07');P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change();
def test(w):
 x=R.rolling(w,min_periods=w).sum().shift(1);sig=-x.sub(x.median(axis=1),axis=0); y=R
 q=[];ns=[]
 for d in sig.index:
  f=sig.loc[d];z=y.loc[d];ok=f.notna()&z.notna()
  if ok.sum()>=8:q.append(spearmanr(f[ok],z[ok]).statistic);ns.append(ok.sum())
 q=pd.Series(q).dropna();return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean().mean()
for w in [2,3,5,10,15,20,30]:print(w,test(w))
