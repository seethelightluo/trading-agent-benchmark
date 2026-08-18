import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index()
c=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index()
v=pd.concat({s:x.volume.astype(float) for s,x in D.items()},axis=1).reindex(c.index)
r=c.pct_change(); lv=np.log(v.replace(0,np.nan)); vz=(lv-lv.rolling(60,min_periods=30).mean())/lv.rolling(60,min_periods=30).std()
# Contrarian return, strengthened by abnormal volume, risk-normalized and lagged.
raw=(-r.rolling(5,min_periods=5).sum())*(1+vz.clip(-1,2).fillna(0)) / r.rolling(20,min_periods=15).std()
f=raw.sub(raw.mean(axis=1),axis=0).shift(1)
print('instruments',len(D),'range',c.index.min().date(),c.index.max().date())
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; z=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q); ns.append(len(a))
 z=pd.Series(z); print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
 if h==10:
  for n in [250,500]:
   q=z.tail(n); print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('panel_coverage',f.notna().mean().mean(),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
