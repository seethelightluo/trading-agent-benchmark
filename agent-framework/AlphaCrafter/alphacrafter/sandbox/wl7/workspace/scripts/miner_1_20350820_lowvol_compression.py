import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); p=p.loc[:'2035-08-20']; r=p.pct_change()
# Stable-risk compression factor: negative recent/long volatility ratio; lower recent risk ranks higher
f=-(r.rolling(20).std()/r.rolling(120).std())
for h in [5,10,20]:
 y=p.shift(-h)/p-1; z=[]; ns=[]
 for d in p.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8: z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic); ns.append(ok.sum())
 z=pd.Series(z); print('H',h,'obs',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4),'period',p.index.min().date(),p.index.max().date())
