import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); px[s]=d.close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2031-12-10']; r=p.pct_change()
# Defensive low-volatility factor: inverse 40d realized volatility, with a mild positive 60d trend overlay.
rv=r.rolling(40,min_periods=30).std(); sig=(1/rv)*(1+0.5*(p/p.shift(60)-1).clip(-1,1))
print('candidate=trend_conditioned_low_vol; dates=%d instruments=%d'%(len(p),len(U)),flush=True)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[];ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):z.append(q);ns.append(len(a));ds.append(dt)
 z=pd.Series(z,index=ds); print('h=%d dates=%d avg_n=%.2f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()),flush=True)
 print('regimes',z.groupby(z.index.year).agg(['mean','count']).round(6).to_dict('index'),flush=True)
print('coverage=%.6f turnover=%.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),flush=True)
