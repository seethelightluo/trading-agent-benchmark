import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-08-12')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change()
# Short-horizon reversal: lagged negative 3-session return, smoothed over 3 observations.
f=(-r.rolling(3,min_periods=3).sum()).shift(1).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-04-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
 if pd.notna(q):rows.append((d,q,int(f.loc[d].notna().sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean())
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index];print('decay',h,np.nanmean(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_3_20320819_short_reversal_signal.csv');x.to_csv('scripts/miner_3_20320819_short_reversal_ic.csv')
