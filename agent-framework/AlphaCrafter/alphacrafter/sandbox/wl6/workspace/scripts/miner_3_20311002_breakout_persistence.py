import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); fs[s]=d.close.astype(float)
p=pd.concat(fs,axis=1).sort_index().loc[:'2031-10-01']; r=p.pct_change()
# Breakout persistence: normalized distance from 60d midpoint, multiplied by directional-day persistence, volatility scaled.
lo=p.rolling(60).min(); hi=p.rolling(60).max(); mid=(lo+hi)/2
loc=(p-mid)/(hi-lo).replace(0,np.nan)
persist=(r.gt(0).rolling(20).mean()-0.5)*2
vol=r.rolling(20).std()*np.sqrt(252)
sig=loc*persist/vol.replace(0,np.nan)
print('candidate=breakout_persistence; instruments=%d last=%s'%(len(U),p.index.max().date()))
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);ns.append(len(a))
 z=pd.Series(z); print('h=%d dates=%d avg_n=%.2f IC=%.8f ICIR=%.6f hit=%.4f'%(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()))
f=p.shift(-10)/p-1; zz=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8: zz.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman')))
z=pd.Series(dict(zz)); print('regimes10',z.groupby(z.index.year).agg(['mean','count']).round(6).to_dict('index'))
print('coverage=%.6f turnover=%.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
