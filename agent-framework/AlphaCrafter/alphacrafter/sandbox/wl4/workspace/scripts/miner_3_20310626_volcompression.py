import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-06-26')
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close
 p[s]=d[d.index<=cutoff]
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# low recent volatility relative to long volatility, lagged; cross-sectional ranks stabilize scale
f=-(r.rolling(20).std()/(r.rolling(120).std()+1e-8)).shift(1)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z));dates.append(dt)
 x=pd.Series(vals);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 fr=p.shift(-10)/p-1; vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals);print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))
print('coverage',f.notna().sum().sum()/p.notna().sum().sum(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
