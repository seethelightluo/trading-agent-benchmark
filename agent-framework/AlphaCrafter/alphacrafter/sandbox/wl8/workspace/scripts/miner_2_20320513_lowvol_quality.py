import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-05-13')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.DataFrame(P).sort_index().ffill(); r=p.pct_change()
# Low-risk quality: inverse lagged 20d volatility, stabilized by positive 60d trend.
v20=r.rolling(20,min_periods=20).std().shift(1); v60=r.rolling(60,min_periods=60).std().shift(1); tr60=r.rolling(60,min_periods=60).sum().shift(1)
f=(-np.log(v20+1e-12)) + .25*(tr60/(v60*np.sqrt(60)+1e-12))
fr=p.shift(-10)/p-1; rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d]; b=fr.loc[d]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=yy.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2031',z['2031']),('2032',z['2032'])]:print(n,q.mean(),q.mean()/q.std(ddof=1))
f.loc[x.index].to_csv('scripts/miner_2_20320513_lowvol_quality_signal.csv');x.to_csv('scripts/miner_2_20320513_lowvol_quality_ic.csv')