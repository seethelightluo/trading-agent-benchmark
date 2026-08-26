import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-23'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
R20=px.pct_change(20)
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
fac=R20-beta.mul(m.rolling(20).sum(),axis=0)
fac=fac.clip(lower=fac.quantile(.05,axis=1),upper=fac.quantile(.95,axis=1),axis=0)
def calc(h,lo=None,hi=None):
 fwd=px.shift(-h)/px-1; out=[]; ns=[]
 idx=fac.index if lo is None else fac.loc[lo:hi].index
 for dt in idx:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(out); ic=a.mean(); ir=ic/(a.std(ddof=1)/np.sqrt(len(a)))
 print('horizon',h,'dates',len(a),'assets_mean',round(np.mean(ns),2),'mean_ic',round(ic,6),'icir',round(ir,6),'hit',round((a>0).mean(),4))
for h in [1,5,10,20]: calc(h)
for label,lo,hi in [('early','2020-01-01','2024-01-01'),('late','2024-01-01','2028-02-23'),('online','2026-07-16','2028-02-23')]:
 print(label); calc(10,lo,hi)
print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'period',px.index.min().date(),px.index.max().date())
