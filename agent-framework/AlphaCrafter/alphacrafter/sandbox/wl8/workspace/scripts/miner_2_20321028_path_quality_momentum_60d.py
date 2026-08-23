import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-10-28')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
# medium-horizon momentum with explicit risk and path-quality adjustment; all inputs lagged one day
ret=r.rolling(60,min_periods=60).sum().shift(1); vol=r.rolling(20,min_periods=20).std().shift(1); path=(r.rolling(60,min_periods=60).mean().shift(1)/r.rolling(60,min_periods=60).std().shift(1)).clip(-5,5)
f=(ret/(vol*np.sqrt(20))).clip(-6,6)*((1+path.clip(-2,2))/2)
f=f.ewm(span=5,min_periods=5).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-21]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
 if pd.notna(q):rows.append((d,q,int(f.loc[d].notna().sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean())
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:print('decay',h,np.nanmean([ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index]))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20321028_path_quality_momentum_60d_signal.csv');x.to_csv('scripts/miner_2_20321028_path_quality_momentum_60d_ic.csv')
