import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 fs[s]=d.drop_duplicates('date').set_index('date').close.astype(float).sort_index()
p=pd.concat(fs,axis=1).sort_index().loc[:'2031-08-06']
r=p.pct_change();
# Momentum normalized by typical absolute daily movement; interpretable range/path risk normalization.
path=r.abs().rolling(20).mean()
sig=p.pct_change(10)/(path*np.sqrt(20))
print('candidate=range_normalized_momentum; universe=',len(fs),'last=',p.index.max().date())
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);ns.append(len(a))
 z=pd.Series(z)
 print(f'h={h} dates={len(z)} avg_n={np.mean(ns):.2f} IC={z.mean():.8f} ICIR={z.mean()/z.std(ddof=1)*np.sqrt(len(z)):.6f} hit={(z>0).mean():.4f}')
# yearly regimes for primary horizon
f=p.shift(-10)/p-1; zz=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8: zz.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman')))
z=pd.Series(dict(zz)); print('regimes10',z.groupby(z.index.year).agg(['mean','count']).to_dict('index'))
print('coverage=',sig.notna().sum(axis=1).mean()/15,'turnover=',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
