import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={};
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); p[a]=d.set_index('date').close
p=pd.DataFrame(p).sort_index(); r=p.pct_change();
# low-vol signal, lagged 20d vol, with a mild 60d trend gate to avoid pure risk proxy
v=r.rolling(20).std().shift(1); trend=p.pct_change(60).shift(1)
sig=-(np.log(v)).rank(axis=1,pct=True) # higher ranks? rank of -log vol gives low vol high
# actually rank row on -v
sig=(-v).rank(axis=1,pct=True)
fwd=p.shift(-1)/p-1
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in p.index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=np.array(z); print(h,len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0))
print('coverage',sig.notna().sum().sum()/sig.size)
