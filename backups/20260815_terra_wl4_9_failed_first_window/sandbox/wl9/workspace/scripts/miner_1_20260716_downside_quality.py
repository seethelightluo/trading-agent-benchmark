import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end].close for s in U}
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# downside-risk quality: assets with less realized loss volatility in the trailing window rank higher
# denominator uses only completed returns through decision date.
down=r.where(r<0,0).rolling(20,min_periods=15).std()
F=-down
# compare against existing factor proxies for redundancy
for h in [1,5,10]:
 y=p.shift(-h)/p-1; a=[];ns=[];ds=[]; years={}
 for d in p.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q);ns.append(len(z));ds.append(d);years.setdefault(str(d.year),[]).append(q)
 a=np.array(a); print('h',h,'obs',len(a),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(v),4) for k,v in years.items()})
# rank turnover, coverage, correlations with factor proxies
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,'coverage',F.notna().sum().sum()/(len(F)*15))
proxies={'rev5':-r.rolling(5).sum(),'mom20':r.rolling(20).sum(),'peer':r.sub(r.median(axis=1),axis=0).rolling(5).sum()}
for k,x in proxies.items(): print('corr',k,F.stack().corr(x.stack()))
# recent slices daily
h=1;y=p.shift(-h)/p-1;a=[];ds=[]
for d in p.index:
 z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):a.append(q);ds.append(d)
a=np.array(a);ds=pd.DatetimeIndex(ds)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print('regime',lo,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
