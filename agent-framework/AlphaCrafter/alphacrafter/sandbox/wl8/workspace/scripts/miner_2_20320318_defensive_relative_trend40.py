import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
defs=['XAU','US10Y','CN10Y']; dr=r[defs].mean(axis=1)
# Slippage-safe 40d relative trend, normalized by lagged 60d volatility.
f=(r.rolling(40,min_periods=40).sum().sub(dr.rolling(40,min_periods=40).sum(),axis=0).div(r.rolling(60,min_periods=60).std().shift(1),axis=0)).shift(1)
rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>pd.Timestamp('2032-03-18'): continue
 a=f.loc[d]; b=(p.iloc[i+10]/p.iloc[i]-1); ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[]
 for i,d in enumerate(p.index[:-h]):
  if d not in x.index: continue
  a=f.loc[d]; b=p.iloc[i+h]/p.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('60',z.tail(60)),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
f.to_csv('scripts/miner_2_20320318_defensive_relative_trend40_signal.csv'); x.to_csv('scripts/miner_2_20320318_defensive_relative_trend40_ic.csv')
