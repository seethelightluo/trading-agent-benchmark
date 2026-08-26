import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-04-19'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); eq=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']; m=r[eq].mean(axis=1)
rows=[]
for i,dt in enumerate(r.index):
 if i<61: continue
 win=r.iloc[i-60:i]; down=win.loc[m.iloc[i-60:i]<0]
 if len(down)<10: continue
 f=-(down.mean()/win.std(ddof=1)).replace([np.inf,-np.inf],np.nan)
 for h in [1,5,10]:
  y=px.shift(-h).loc[dt]/px.loc[dt]-1; z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',px.index.min().date(),end.date(),'dates',len(r),'instruments',len(S),'observations',len(A),'mean coverage',round(A.n.mean()/15,4))
for h,g in A.groupby('h'):
 def stats(q): return (len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4),round(q.n.mean(),2)) if len(q) else None
 print('horizon',h,'all(obs,IC,ICIR,hit,n)',stats(g),'online',stats(g[g.date>=pd.Timestamp('2026-07-16')]),'recent',stats(g[g.date>=pd.Timestamp('2027-04-20')]))
