import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def c(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
P=pd.DataFrame({a:c(a) for a in A}).sort_index(); lr=np.log(P).diff()
# Candidate: market-beta-neutral residual momentum, volatility scaled.
mkt=lr.mean(axis=1)
beta=lr.rolling(60,min_periods=40).cov(mkt).div(mkt.rolling(60,min_periods=40).var(),axis=0)
r15=lr.rolling(15,min_periods=12).sum(); rm15=mkt.rolling(15,min_periods=12).sum()
res=r15-beta.mul(rm15,axis=0)
v40=lr.rolling(40,min_periods=30).std()*np.sqrt(252)
sig=(-res/v40).shift(1)
print('END',P.index.max(),'assets',len(A),'coverage_cells',sig.notna().mean().mean(),'valid_dates',sig.notna().all(axis=1).sum())
def run(s,h,lo=None,hi=None):
 f=P.shift(-h)/P-1; vals=[]; ns=[]
 ix=s.index if lo is None else s.loc[lo:hi].index
 for d in ix:
  z=pd.concat([s.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 x=np.array(vals); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
for h in [1,5,10,20]: print('H',h,'n meanN IC ICIR hit',run(sig,h))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-12-31')]: print('REG10',lo,run(sig,10,lo,hi))
r=sig.rank(axis=1,pct=True); print('turn10',np.nanmean((r-r.shift(10)).abs().sum(axis=1)/r.notna().sum(axis=1)))
print('mean_cs_std',sig.std(axis=1).mean())
