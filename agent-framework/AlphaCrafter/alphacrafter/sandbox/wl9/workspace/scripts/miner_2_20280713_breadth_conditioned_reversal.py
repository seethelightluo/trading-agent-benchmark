import numpy as np,pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-07-12'); P={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.astype(float); P[s]=x.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); eq=U[:8]; rows=[]
for i in range(25,len(px)-10):
 dt=px.index[i]; hist=r.iloc[i-20:i]; breadth=(hist[eq]>0).mean(axis=1).mean()
 ret=px.iloc[i-1].div(px.iloc[i-6])-1; vol=hist.std().replace(0,np.nan)
 f=(-ret/vol) if breadth<0.5 else (ret/vol)
 for h in [1,5,10]:
  y=px.iloc[i+h].div(px.iloc[i])-1; z=pd.concat([f,y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.y).statistic))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',px.index.min().date(),end.date(),'dates',len(px),'instruments',len(U),'observations',len(A),'coverage',round(A.n.mean()/15,4))
for h,g in A.groupby('h'):
 print('horizon',h,'dates',len(g),'mean_n',round(g.n.mean(),2),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6),'hit',round((g.ic>0).mean(),4))
 for name,cut in [('online','2026-07-16'),('recent','2027-07-13'),('ytd','2028-01-01')]:
  q=g[g.date>=pd.Timestamp(cut)]; print(name,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
