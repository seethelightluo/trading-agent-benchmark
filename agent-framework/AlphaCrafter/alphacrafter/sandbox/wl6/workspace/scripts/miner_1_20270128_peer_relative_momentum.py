import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().loc[:'2027-01-28']; r=p.pct_change(); ret5=p.pct_change(5); vol20=r.rolling(20).std()
f=ret5.sub(ret5.median(axis=1),axis=0).div(vol20,axis=0)
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).dropna().mean()
for h in [1,3,5,10]:
 fr=p.shift(-h).div(p)-1; rows=[]
 for i in range(21,len(p)-h):
  sig=f.iloc[i-1]; y=fr.iloc[i-1]; z=pd.concat([sig,y],axis=1).dropna()
  if len(z)>=8: rows.append((p.index[i-1],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(rows,columns=['date','n','ic']); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)*np.sqrt(252)
 print(f'h={h} dates={len(q)} avg_n={q.n.mean():.2f} coverage={q.n.mean()/15:.4f} IC={ic:.5f} ICIR={ir:.5f} hit={(q.ic>0).mean():.4f} turnover={turnover:.5f}')
 for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-01-28')]:
  x=q[(q.date>=pd.Timestamp(a))&(q.date<=pd.Timestamp(b))].ic
  print(' ',lab,'dates',len(x),'IC',round(x.mean(),5) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),5) if len(x)>1 else None)
