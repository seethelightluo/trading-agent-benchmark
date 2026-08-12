import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 px[s]=d
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in px.items() if d is not None}).sort_index()
ret=cl.pct_change(); r20=cl.pct_change(20); v20=ret.rolling(20).std(); v60=ret.rolling(60).std()
raw=(r20/v60)*(1+0.7*np.clip(1-v20/v60,-1,1)); sig=raw.shift(1)
rows=[]
for h in [1,5,10,20]:
 fr=cl.shift(-h)/cl-1; ics=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=pd.Series(ics).dropna(); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'dates',len(a),'avgN',np.mean(ns))
valid=sig.notna().sum(axis=1)/len(U); turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('coverage',valid.mean(),'turnover',turn,'endpoint',cl.index[-1].date())
fr=cl.shift(-20)/cl-1
for yr in sorted(set(sig.index.year)):
 vals=[]
 for dt in sig.index[sig.index.year==yr]:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if vals: print('yr',yr,'ic',np.nanmean(vals),'n',len(vals))
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20310109_vol_compression_trend_signal.csv')
