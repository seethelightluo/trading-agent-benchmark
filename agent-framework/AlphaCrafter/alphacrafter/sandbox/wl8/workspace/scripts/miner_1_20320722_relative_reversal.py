import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-07-22')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); x=r.shift(1).rolling(5,min_periods=5).sum(); f=-(x.sub(x.mean(axis=1),axis=0)).rolling(3,min_periods=3).mean()
def ic(a,b):
 o=a.notna()&b.notna()
 return spearmanr(a[o],b[o]).statistic if o.sum()>=8 and a[o].nunique()>2 else np.nan
fr={h:p.shift(-h)/p-1 for h in [1,5,10,20]};rows=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-08-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],fr[10].loc[d]);
 if pd.notna(q):rows.append((d,q,(f.loc[d].notna()&fr[10].loc[d].notna()).sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').ic
print('dates',len(z),'start',z.index.min().date(),'end',z.index.max().date(),'avg_n',15,'coverage',f.loc[z.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:print('decay',h,np.nanmean([ic(f.loc[d],fr[h].loc[d]) for d in z.index]))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[z.index].to_csv('scripts/miner_1_20320722_relative_reversal_signal.csv');z.to_csv('scripts/miner_1_20320722_relative_reversal_ic.csv')
