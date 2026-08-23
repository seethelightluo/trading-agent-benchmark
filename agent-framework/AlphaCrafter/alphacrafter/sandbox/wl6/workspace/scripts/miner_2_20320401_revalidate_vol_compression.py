import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-03-31')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); px[s]=d.close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:cut]; r=p.pct_change(); s10=r.rolling(10,min_periods=10).std(); s60=r.rolling(60,min_periods=40).std(); sig=-(p/p.shift(15)-1)*np.clip(s60/(s10+1e-12),.25,4.)
print('revalidation 15/60 cutoff',cut.date(),'dates',len(p),'instruments',len(U))
for start in [pd.Timestamp('2020-01-01'),pd.Timestamp('2028-01-01'),pd.Timestamp('2030-01-01'),pd.Timestamp('2031-01-01'),pd.Timestamp('2032-01-01')]:
 f=p.shift(-20)/p-1; z=[]; n=[]; ds=[]
 for dt in sig.index:
  if dt<start: continue
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): z.append(q);n.append(len(a));ds.append(dt)
 z=pd.Series(z,index=ds); print('start=%s dates=%d avg_n=%.2f cov=%.4f IC=%+.6f ICIR=%+.4f hit=%.4f'%(start.date(),len(z),np.mean(n),np.mean(n)/15,z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean()))
print('turnover=%.6f'%sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
