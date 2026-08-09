import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-03-25'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); P[s]=d.close.loc[:end]
P=pd.DataFrame(P).ffill(); y=P.shift(-1)/P-1
for w in [3,4,5,6,8,10]:
 f=P.pct_change(20)/4-P.pct_change(w); q=[]; ns=[]
 for dt in f.index:
  x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],z[ok]).statistic);ns.append(ok.sum())
 q=np.array(q); print(w,len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'N',np.mean(ns))
