import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; px={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); rv=r.rolling(20,min_periods=15).std(); skew=r.rolling(60,min_periods=40).skew()
# Favor reversal after downside-skewed, volatile paths; lag every input one day.
f=(-r.rolling(10,min_periods=10).sum().shift(1)/(rv.shift(1)*np.sqrt(20)))*(1+0.25*skew.shift(1).clip(-2,2))
for h in [1,5,10,20]:
 z=[]; ns=[]
 for d in f.index:
  x=f.loc[d]; y=(P.shift(-h)/P-1).loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q):z.append(q);ns.append(ok.sum())
 s=pd.Series(z); print('horizon',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
