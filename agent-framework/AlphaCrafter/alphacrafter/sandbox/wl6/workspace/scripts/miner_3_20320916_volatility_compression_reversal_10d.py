import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-09-15']; r=p.pct_change()
v10=r.rolling(10).std(); v60=r.rolling(60).std()
# compression (low short vol relative to long vol), direction reversal via recent return sign
sig=-(p.pct_change(10))*v10.div(v60).replace(0,np.nan)
f=p.shift(-10).div(p)-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=np.array(rows)
print('candidate=negative 10d return * short/long volatility ratio; horizon 10d')
print('dates',len(a),'avg instruments',sig.notna().sum(axis=1).mean(),'coverage',sig.notna().sum().sum()/sig.size)
print('IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20,40]:
 ff=p.shift(-h).div(p)-1; q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
rr=pd.Series(a,index=sig.index[-len(a):]); print('year_IC');print(rr.groupby(rr.index.year).mean().to_string())
