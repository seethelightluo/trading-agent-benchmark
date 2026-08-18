import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-10-05')
p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'];p[s]=d[d.index<=end]
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); med=r.median(axis=1)
# relative 5d reversal: recent asset return relative to cross-asset median, invert
for w in [3,5,10]:
 rel=(p.pct_change(w).sub(p.pct_change(w).median(axis=1),axis=0))*-1
 y=r.shift(-1); a=[]; c=[]
 for dt in rel.index:
  z=pd.concat([rel.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);c.append(len(z)/15)
 a=np.array(a); print('w',w,'n',len(a),'cov',np.mean(c),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 print('early/late',a[:len(a)//2].mean(),a[len(a)//2:].mean())
