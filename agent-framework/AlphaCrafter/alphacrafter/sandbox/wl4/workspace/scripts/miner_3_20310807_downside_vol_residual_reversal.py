import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-08-07')
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close
 p[s]=d[d.index<=cutoff]
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# Downside-volatility scaled cross-sectional residual reversal.
# Negative returns receive extra risk scaling; signal is lagged one session.
L,V=10,40
cs=r.rolling(L).sum(); resid=cs.sub(cs.median(axis=1),axis=0)
down=r.where(r<0,0).rolling(V,min_periods=20).std()
allvol=r.rolling(V,min_periods=20).std()
# blend downside and total volatility to avoid unstable zero downside estimates
vol=(0.7*down+0.3*allvol)
f=(-resid/(vol+1e-8)).shift(1)
fr={h:p.shift(-h)/p-1 for h in [5,10,20]}
for h in [5,10,20]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))
print('coverage',round(f.notna().sum().sum()/p.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'instruments',len(U),'dates',len(p))
