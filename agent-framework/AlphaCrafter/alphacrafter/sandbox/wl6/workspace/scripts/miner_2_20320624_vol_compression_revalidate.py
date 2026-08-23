import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-06-23')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}; p=pd.concat(p,axis=1).sort_index().loc[:cut]; r=p.pct_change(); s10=r.rolling(10,min_periods=10).std(); s60=r.rolling(60,min_periods=40).std(); sig=-(p/p.shift(15)-1)*np.clip(s60/(s10+1e-12),.25,4.)
print('cutoff',p.index.max().date(),'dates',len(p),'instruments',len(U))
for h in [5,10,20]:
 f=p.shift(-h)/p-1; z=[];n=[];ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):z.append(q);n.append(len(a));ds.append(dt)
 z=np.array(z); print({'horizon':h,'valid_dates':len(z),'avg_instruments':round(np.mean(n),3),'coverage':round(np.mean(n)/15,4),'IC':round(z.mean(),6),'ICIR':round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6),'hit':round(np.mean(z>0),4)})
 if h==20: print('regimes',pd.DataFrame({'ic':z},index=ds).groupby(lambda x:x.year).ic.mean().round(6).to_dict())
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
