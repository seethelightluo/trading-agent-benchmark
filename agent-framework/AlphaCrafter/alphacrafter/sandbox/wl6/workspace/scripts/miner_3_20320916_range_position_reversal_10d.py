import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv');d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-09-15']; r=p.pct_change()
# location of current close within 60d return range, low position expected rebound
lo=p.rolling(60).min(); hi=p.rolling(60).max(); loc=(p-lo)/(hi-lo).replace(0,np.nan)
sig=0.5-loc
f=p.shift(-10).div(p)-1
rows=[]; dates=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt)
a=np.array(rows);print('candidate=60d range-position reversal; horizon 10d');print('dates',len(a),'avg_n',sig.notna().sum(axis=1).mean(),'coverage',sig.notna().sum().sum()/sig.size);print('IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20,40]:
 ff=p.shift(-h).div(p)-1;q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
rr=pd.Series(a,index=dates);print('year_IC');print(rr.groupby(rr.index.year).mean().to_string())
