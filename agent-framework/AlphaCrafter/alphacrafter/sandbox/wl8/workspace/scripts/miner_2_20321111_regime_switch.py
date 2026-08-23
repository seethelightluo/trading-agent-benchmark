import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-11-10')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change();res=r-r.mean(axis=1).values[:,None]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill(); high=(v.shift(1)>v.rolling(252,min_periods=120).median().shift(1)).astype(float)
rev=(-res.rolling(5,min_periods=5).sum().shift(1)/res.rolling(20,min_periods=20).std().shift(1)).clip(-4,4)
trend=(res.rolling(10,min_periods=10).sum().shift(1)/res.rolling(30,min_periods=30).std().shift(1)).clip(-4,4)
f=rev.mul(high.values,axis=0)+trend.mul((1-high).values,axis=0);f=f.ewm(span=3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8:return np.nan
 return spearmanr(a[ok],b[ok]).statistic if a[ok].nunique()>2 else np.nan
rows=[]
for i,d in enumerate(p.index[:-21]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.iloc[i],(p.shift(-10)/p-1).iloc[i]);
 if pd.notna(q):rows.append((d,q,int(f.iloc[i].notna().sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean())
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:print('decay',h,np.nanmean([ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index]))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20321111_regime_switch_signal.csv');x.to_csv('scripts/miner_2_20321111_regime_switch_ic.csv')
