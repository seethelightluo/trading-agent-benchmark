import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b='../persistent/stock_data'
p={s:pd.read_csv(os.path.join(b,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.DataFrame(p).sort_index(); p=p.loc[:'2032-02-18']; r=p.pct_change()
# Short-window downside-risk contrarian: lagged 20d return divided by lagged 20d downside deviation.
ret=p.shift(1)/p.shift(21)-1
down=r.where(r<0).rolling(20,min_periods=6).std().shift(1)*np.sqrt(252)
f=(ret/down).replace([np.inf,-np.inf],np.nan)
print('cutoff',p.index.max().date(),'assets',len(U),'factor coverage',round(f.notna().sum(axis=1).mean()/15,4))
for h in [5,10,20]:
 y=p.shift(-h)/p-1; a=[]; ns=[]; dates=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q); ns.append(len(z)); dates.append(d)
 a=np.asarray(a); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(len(a))
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,4),'hit',round(np.mean(a>0),4),'turnover_proxy',round(np.mean(np.diff(np.argsort(np.argsort(f.fillna(0),axis=1),axis=1),axis=0)!=0),4) if False else 'NA')
 for n in [365,730,1095]:
  q=a[-n:] if len(a)>=n else a
  print(' recent',n,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),4))
