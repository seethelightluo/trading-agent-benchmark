import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 fs[s]=d.close.astype(float)
p=pd.concat(fs,axis=1).sort_index().loc[:'2031-09-03']
lo=p.rolling(40).min(); hi=p.rolling(40).max()
loc=(p-lo)/(hi-lo).replace(0,np.nan)
sig=loc-loc.shift(5)
print('factor=range_location_acceleration; dates=%d instruments=%d last=%s'%(len(p),len(U),p.index.max().date()))
for h in [5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(len(a))
 z=pd.Series(vals); print('h=%d dates=%d avg_n=%.2f IC=%.8f ICIR=%.6f hit=%.4f'%(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()))
f=p.shift(-10)/p-1; vals=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8: vals.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman')))
z=pd.Series(dict(vals)); print('regimes10',z.groupby(z.index.year).mean().round(6).to_dict())
print('coverage=%.6f turnover=%.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
