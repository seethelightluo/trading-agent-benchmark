import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; eq=S[:8]; end=pd.Timestamp('2028-05-17'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r[eq].mean(axis=1); rows=[]
for i,dt in enumerate(r.index):
 if i<65+10: continue
 fs=[]
 for W in [20,60]:
  win=r.iloc[i-W:i]; down=win.loc[m.iloc[i-W:i]<0]
  if len(down)<5: fs.append(pd.Series(index=S,dtype=float)); continue
  fs.append(-(down.mean()/win.std(ddof=1)).replace([np.inf,-np.inf],np.nan).rank(pct=True))
 f=(fs[0]+fs[1])/2
 for h in [1,5,10]:
  y=px.shift(-h).loc[dt]/px.loc[dt]-1; z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',px.index.min().date(),end.date(),'instruments',len(S),'observations',len(A),'coverage',round(A.n.mean()/15,4))
for h,g in A.groupby('h'):
 def st(q): return (len(q),round(q.n.mean(),2),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
 print('horizon',h,'all',st(g),'online',st(g[g.date>=pd.Timestamp('2026-07-16')]),'recent',st(g[g.date>=pd.Timestamp('2027-05-18')]))
