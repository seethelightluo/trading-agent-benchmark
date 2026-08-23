import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-02-18')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# Regime-conditioned momentum: lagged 20d momentum, reversed during elevated VIX (above trailing 60d median).
hi=(v>v.rolling(60,min_periods=60).median()).astype(float)
base=r.rolling(20,min_periods=20).sum().div(r.rolling(20,min_periods=20).std().shift(1),axis=0)
f=(base*(1-2*hi)).shift(1)
fr=p.shift(-10)/p-1; rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d];b=fr.loc[d];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum(),hi.loc[d]))
x=pd.DataFrame(rows,columns=['date','ic','n','highvix']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=yy.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('60',z.tail(60)),('2029',z['2029']),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
print('regimes',x.groupby('highvix').ic.agg(['mean','count']))
f.to_csv('scripts/miner_1_20320219_vix_conditioned_momentum_signal.csv');x.to_csv('scripts/miner_1_20320219_vix_conditioned_momentum_ic.csv')
