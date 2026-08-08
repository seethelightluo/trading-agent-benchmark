import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}; idx=sorted(set.intersection(*[set(x.index) for x in D.values()]))
for w in [2,3,5,10]:
 out=[]
 for t in idx:
  v=[];f=[]
  for a in A:
   x=D[a];k=x.index.get_loc(t)
   if k<w+1 or k+1>=len(x):continue
   r=x.close.pct_change(w).iloc[k]; fr=x.close.iloc[k+1]/x.close.iloc[k]-1
   if np.isfinite(r) and np.isfinite(fr):v.append(-r);f.append(fr)
  if len(v)>=8:
   q=spearmanr(v,f).statistic
   if np.isfinite(q):out.append((t,q,len(v)))
 r=pd.DataFrame(out,columns=['d','ic','n']).set_index('d'); z=r.ic
 print('W',w,'dates',len(r),'N',r.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'latest120',z.tail(120).mean(),z.tail(120).mean()/z.tail(120).std(ddof=1))
PY