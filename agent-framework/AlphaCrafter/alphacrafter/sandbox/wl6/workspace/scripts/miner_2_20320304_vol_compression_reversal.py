import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
px={s:(x.set_index('date')['close'].astype(float) if x is not None else pd.Series(dtype=float)) for s,x in D.items()}
rets=pd.DataFrame({s:p.pct_change() for s,p in px.items()}); vol10=rets.rolling(10).std(); vol60=rets.rolling(60).std(); r10=pd.DataFrame({s:p.pct_change(10) for s,p in px.items()})
f=(-r10*(vol60/(vol10+1e-8)).clip(0.25,4.0)).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20]:
 fw=pd.DataFrame({s:p.shift(-h)/p-1 for s,p in px.items()}); vals=[]; counts=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); counts.append(len(z))
 a=np.asarray(vals); print('h',h,'dates',len(a),'avg_n',round(float(np.mean(counts)),2),'coverage',round(float(np.mean(counts)/15),4),'IC',round(float(np.nanmean(a)),8),'ICIR',round(float(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(len(a))),4),'hit',round(float(np.mean(a>0)),4))
r=f.rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean().mean()),'date_min',str(f.index.min().date()),'date_max',str(f.index.max().date()))
