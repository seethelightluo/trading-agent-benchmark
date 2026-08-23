import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-10-27']; r=p.pct_change()
v10=r.rolling(10).std(); v60=r.rolling(60).std()
sig=-(p.pct_change(10))*v10.div(v60).replace(0,np.nan)
rows={h:[] for h in [5,10,20,40]}; dates={h:[] for h in rows}
for dt in sig.index:
 for h in rows:
  f=p.shift(-h).div(p)-1; z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates[h].append(dt)
print('candidate=negative 10d return * short/long volatility ratio; cutoff=2032-10-27')
print('dates/instruments/coverage',len(dates[10]),round(sig.notna().sum(axis=1).mean(),2),round(sig.notna().sum().sum()/sig.size,4))
for h in rows:
 a=np.array(rows[h]); print('horizon',h,'n',len(a),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a=np.array(rows[10]); rr=pd.Series(a,index=dates[10]); print('year_IC');print(rr.groupby(rr.index.year).mean().round(6).to_string())
