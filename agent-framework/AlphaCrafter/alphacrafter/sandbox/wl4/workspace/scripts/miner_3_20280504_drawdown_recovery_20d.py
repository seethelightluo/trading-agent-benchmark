import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; px={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
low=P.rolling(60,min_periods=40).min(); rebound=P/low-1
allv=r.rolling(20,min_periods=15).std(); down=np.minimum(r,0.0); downv=down.rolling(20,min_periods=15).std()
f=(rebound/(allv*np.sqrt(20))*(1-downv/(allv+1e-12))).shift(1)
for h in [1,5,10,20]:
 z=[]; ns=[]; dates=[]
 for d in f.index:
  x=f.loc[d]; y=(P.shift(-h)/P-1).loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q):z.append(q);ns.append(ok.sum());dates.append(d)
 s=pd.Series(z,index=dates); recent=s.tail(250)
 print('horizon',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(recent.mean(),6),round(recent.mean()/recent.std(ddof=1),6))
