import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-15'); p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change();v=r.rolling(30,min_periods=30).std().shift(1)
# 3-day mean-reversion shock normalized by 30d vol, with lagged 10d cross-sectional median removal
raw=-r.rolling(3,min_periods=3).sum().shift(1)/(v*np.sqrt(3)+1e-12)
f=raw.sub(raw.median(axis=1),axis=0).rolling(2,min_periods=2).mean().shift(1);fr=p.shift(-10)/p-1
rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 a=f.loc[d];b=fr.loc[d];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3:rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=y.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_3_20320429_3d_reversal_signal.csv');x.to_csv('scripts/miner_3_20320429_3d_reversal_ic.csv')
