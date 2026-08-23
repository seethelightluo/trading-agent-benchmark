import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); d=r[['XAU','US10Y','CN10Y']].mean(axis=1)
# Faster defensive-relative trend with lagged volatility normalization.
f=(r.rolling(25,min_periods=25).sum().sub(d.rolling(25,min_periods=25).sum(),axis=0).div(r.rolling(50,min_periods=50).std().shift(1),axis=0)).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-10]):
 if dt<pd.Timestamp('2020-06-01') or p.index[i+10]>pd.Timestamp('2032-03-18'): continue
 a=f.iloc[i]; b=p.iloc[i+10]/p.iloc[i]-1; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[]
 for i,dt in enumerate(p.index[:-h]):
  if dt not in x.index: continue
  a=f.iloc[i]; b=p.iloc[i+h]/p.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('60',z.tail(60)),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std())
f.to_csv('scripts/miner_2_20320318_defensive_relative_trend25_signal.csv'); x.to_csv('scripts/miner_2_20320318_defensive_relative_trend25_ic.csv')
