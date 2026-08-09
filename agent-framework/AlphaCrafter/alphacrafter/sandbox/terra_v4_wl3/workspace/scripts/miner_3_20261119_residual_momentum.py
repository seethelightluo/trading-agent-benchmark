import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close;P[s]=d
P=pd.DataFrame(P).sort_index().loc[:'2026-11-18'];R=P.pct_change(fill_method=None)
# Robust cross-asset residual momentum: 20d return minus trailing beta to equal-weight market times market return.
m=R.mean(axis=1); cov=R.rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
raw=R.rolling(20,min_periods=20).sum(); market=m.rolling(20,min_periods=20).sum();F=raw-beta.mul(market,axis=0)
F.to_csv('scripts/miner_3_20261119_residual_momentum_signal.csv',index_label='date')
print('assets',len(U),'rows',len(F),'period',F.index.min(),F.index.max())
for h in [1,5,10]:
 Y=P.pct_change(h,fill_method=None).shift(-h);vals=[];ns=[];ds=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:print('regimes',[(y,round(s[s.index.year==y].mean(),5),len(s[s.index.year==y])) for y in range(2020,2027)])
print('coverage',F.notna().sum().sum()/(len(F)*15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
