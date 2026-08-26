import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-05-03'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.astype(float); D[s]=x.loc[:end]
px=pd.DataFrame(D).sort_index(); R=px.pct_change(); m=R[U[:8]].mean(axis=1); W=120; out=[]
for i in range(W+1,len(R)-10):
 hist=R.iloc[i-W:i]; mm=m.iloc[i-W:i]; mask=mm<0
 if mask.sum()<20: continue
 vals={}
 for s in U:
  x=hist[s]; ok=mask & x.notna() & mm.notna()
  if ok.sum()<20 or np.var(mm[ok])==0: continue
  vals[s]=-np.cov(x[ok],mm[ok],ddof=1)[0,1]/np.var(mm[ok],ddof=1)
 f=pd.Series(vals)
 for h in [1,5,10]:
  y=px.iloc[i+h].div(px.iloc[i])-1; q=pd.concat([f,y.rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((R.index[i],h,len(q),spearmanr(q.iloc[:,0],q.y).statistic))
A=pd.DataFrame(out,columns=['date','h','n','ic']); print('period',px.index.min().date(),end.date(),'dates',len(R),'instruments',len(D),'observations',len(A),'coverage',round(A.n.mean()/15,4))
for h,g in A.groupby('h'):
 print('horizon',h,'dates',len(g),'mean_n',round(g.n.mean(),2),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6),'hit',round((g.ic>0).mean(),4))
 for name,cut in [('online','2026-07-16'),('recent','2027-05-04')]:
  q=g[g.date>=pd.Timestamp(cut)]; print(name,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
