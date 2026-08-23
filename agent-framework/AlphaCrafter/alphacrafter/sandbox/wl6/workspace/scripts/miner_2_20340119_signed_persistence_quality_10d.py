import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-01-18'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 px[s]=d.close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:cut]; r=p.pct_change()
# Signed persistence quality: trend magnitude, adjusted for path persistence and realized risk.
# Positive-day excess over negative-day frequency rewards smooth directional trends.
up=(r>0).rolling(20,min_periods=15).mean(); dn=(r<0).rolling(20,min_periods=15).mean()
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
sig=(p/p.shift(20)-1)*(up-dn)/(vol+1e-12)
print('candidate=signed_persistence_quality_10d dates=%d instruments=%d first=%s last=%s'%(len(p),len(U),p.index.min().date(),p.index.max().date()),flush=True)
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1; z=[]; ns=[]; ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q); ns.append(len(a)); ds.append(dt)
 z=pd.Series(z,index=ds); ir=z.mean()/z.std(ddof=1)*np.sqrt(len(z))
 print('h=%d dates=%d avg_n=%.2f coverage=%.5f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(ns),np.mean(ns)/15,z.mean(),ir,(z>0).mean()),flush=True)
 print('regimes',z.groupby(z.index.year).mean().round(6).to_dict(),flush=True)
print('turnover=%.6f'%sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),flush=True)
