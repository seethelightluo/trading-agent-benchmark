import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-10-27']; r=p.pct_change(); ratio=r.rolling(10).std().div(r.rolling(60).std()).replace(0,np.nan)
sig=p.pct_change(10)*ratio
for h in [5,10,20,40]:
 out=[]; ds=[]
 f=p.shift(-h).div(p)-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt)
 a=np.array(out); print(h,len(a),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h==10: print(pd.Series(a,index=ds).groupby(pd.Series(ds).dt.year).mean().round(6).to_string())
print('coverage',sig.notna().sum().sum()/sig.size,'avg_n',sig.notna().sum(axis=1).mean())
