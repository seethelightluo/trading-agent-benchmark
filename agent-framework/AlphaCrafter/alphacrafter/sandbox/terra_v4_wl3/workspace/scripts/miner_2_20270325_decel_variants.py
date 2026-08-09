import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-03-25'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); P[s]=d.close.loc[:end]
P=pd.DataFrame(P).ffill(); R=P.pct_change(); V=R.rolling(20).std()*np.sqrt(252); F=P.pct_change(20)/4
for w in [2,3,4,5,7,10]:
 f=(F-P.pct_change(w))/V; y=P.shift(-1)/P-1; q=[]
 for dt in f.index:
  x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],z[ok]).statistic)
 q=np.array(q); print(w,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
