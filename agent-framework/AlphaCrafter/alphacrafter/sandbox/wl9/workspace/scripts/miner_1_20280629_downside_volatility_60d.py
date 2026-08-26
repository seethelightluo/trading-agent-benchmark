import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-06-28'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); rows=[]
for i,dt in enumerate(r.index):
 if i<61: continue
 w=r.iloc[i-60:i]
 up=w.where(w>0).std(ddof=1); dn=(-w.where(w<0)).std(ddof=1)
 f=(dn-up)/(up+dn)
 for h in [1,5,10,20]:
  y=px.shift(-h).loc[dt]/px.loc[dt]-1; z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',px.index.min().date(),end.date(),'instruments',len(S),'observations',len(A),'coverage',round(A.n.mean()/15,4))
def st(q):
 return {'dates':len(q),'mean_n':round(q.n.mean(),2),'IC':round(q.ic.mean(),6),'ICIR':round(q.ic.mean()/q.ic.std(ddof=1),6),'hit':round((q.ic>0).mean(),4)}
for h,g in A.groupby('h'):
 print('horizon',h,'all',st(g),'online',st(g[g.date>=pd.Timestamp('2026-07-16')]),'recent252',st(g[g.date>=pd.Timestamp('2027-06-15')]))
