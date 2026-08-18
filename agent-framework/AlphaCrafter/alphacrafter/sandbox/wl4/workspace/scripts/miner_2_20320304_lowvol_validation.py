import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.concat(D,axis=1).sort_index().loc[:'2032-03-03']; r=np.log(px).diff()
# volatility score: inverse lagged 40d vol, plus stability of vol (inverse vol-of-vol), equal simple interpretable blend
v=r.rolling(40,min_periods=25).std(); vv=v.diff().rolling(20,min_periods=12).std(); f=(0.7*(-v)+0.3*(-vv)).shift(1)
for h in [5,10,20]:
 y=np.log(px.shift(-h)/px); a=[]; ns=[]; tr=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  q=f.loc[dt].rank(pct=True)
  if prev is not None:
   w=pd.concat([q,prev],axis=1).dropna(); tr.append(abs(w.iloc[:,0]-w.iloc[:,1]).mean())
  prev=q
 a=np.array(a); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={np.mean(a):.6f} ICIR={np.mean(a)/(np.std(a,ddof=1)/np.sqrt(len(a))):.4f} hit={np.mean(a>0):.4f} turnover={np.mean(tr):.4f}')
 if h==10:
  for n in [260,520,780]:
   q=a[-n:];print('recent',n,np.mean(q),np.mean(q)/(np.std(q,ddof=1)/np.sqrt(len(q))),np.mean(q>0))
print('cutoff',px.index.max().date(),'assets',px.shape[1])
