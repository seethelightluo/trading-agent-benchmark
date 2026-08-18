import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p={s:pd.read_csv(os.path.join(b,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U};p=pd.DataFrame(p).sort_index();r=p.pct_change()
ret=p.shift(1)/p.shift(41)-1
down=r.where(r<0).rolling(40,min_periods=8).std().shift(1)*np.sqrt(252)
f=(ret/down).replace([np.inf,-np.inf],np.nan)
for h in [10,20]:
 y=p.shift(-h)/p-1; a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(len(a)); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',ic,'ICIR',ir,'hit',np.mean(a>0),'coverage',f.notna().sum(axis=1).mean()/15)
 for n in [365,730,1095]:
  q=a[-n:]; print('recent',n,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
