import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-23')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:end] for s in S}).sort_index()
r=px.pct_change(); rf=r.fillna(0); m=rf.mean(axis=1)
beta=rf.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=px.pct_change(5)-beta.mul(m.rolling(5).sum(),axis=0); fac=-res
# Only retain values with valid price and beta; winsorize within date
fac=fac.where(px.notna()).clip(lower=fac.quantile(.05,axis=1),upper=fac.quantile(.95,axis=1),axis=0)
def run(h,lo=None,hi=None):
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in fac.index if lo is None else fac.loc[lo:hi].index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); ic=a.mean(); ir=ic/(a.std(ddof=1)/np.sqrt(len(a)))
 print(h,len(a),round(np.mean(ns),2),round(ic,6),round(ir,6),round((a>0).mean(),4))
for h in [1,3,5,10,20]: run(h)
for lab,lo,hi in [('early','2020','2023'),('mid','2023','2026'),('online','2026-07-16','2028-02-23')]: print(lab);run(1,lo,hi);run(5,lo,hi)
print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'dates',len(fac))
