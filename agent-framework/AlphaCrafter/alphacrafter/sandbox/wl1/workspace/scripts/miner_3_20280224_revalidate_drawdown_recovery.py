import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2028-02-23')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
acc=r.rolling(20,min_periods=20).sum()-r.rolling(60,min_periods=60).sum()/3
vol=r.rolling(60,min_periods=40).std()*np.sqrt(60); peak=px.rolling(60,min_periods=40).max(); dd=(px/peak-1).clip(upper=0)
f=((acc/(vol+0.005))*(1+dd.rolling(20,min_periods=10).mean())).shift(1)
print('revalidation drawdown_recovery_acceleration cutoff',px.index.max().date(),'dates',len(px))
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.array(I); ds=pd.DatetimeIndex(ds); print('h',h,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'hit',round(np.mean(a>0),4))
 for lab,m in [('2027+',ds>=pd.Timestamp('2027-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01'))]:
  z=a[m]; print(lab,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'dates',int(m.sum()))
print('turnover',round((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6))
