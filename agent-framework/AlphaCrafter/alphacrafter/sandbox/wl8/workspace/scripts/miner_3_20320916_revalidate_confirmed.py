import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-09-09')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change(); mom=r.rolling(20,min_periods=20).sum().shift(1);vol=r.rolling(30,min_periods=30).std().shift(1)*np.sqrt(20); conf=r.rolling(5,min_periods=5).sum().shift(1);f=mom/vol*np.where(conf>0,1,.35);f=pd.DataFrame(f,index=p.index,columns=p.columns)
z=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 a=f.loc[d];b=(p.shift(-10)/p-1).loc[d];ok=a.notna()&b.notna();
 if ok.sum()>=8:z.append((d,spearmanr(a[ok],b[ok]).statistic))
x=pd.DataFrame(z,columns=['date','ic']).set_index('date');q=x.ic
print('dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'coverage',f.loc[x.index].notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for label,s in [('365',q.tail(365)),('180',q.tail(180)),('2032',q[q.index.year==2032])]:print(label,len(s),s.mean(),s.mean()/s.std(ddof=1))
