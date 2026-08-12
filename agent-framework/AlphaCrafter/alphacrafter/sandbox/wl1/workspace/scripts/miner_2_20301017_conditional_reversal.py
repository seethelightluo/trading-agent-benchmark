import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2030-10-16');b=Path('../persistent/stock_data')
D={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut] for s in U};idx=sorted(set().union(*[set(x.index) for x in D.values()]));p=pd.DataFrame({s:x.reindex(idx) for s,x in D.items()}).ffill();r=p.pct_change()
# Conditional short-horizon reversal: reward a 3d pullback only when the
# asset remains above its 60d trend, scaled by recent volatility and lagged.
rev=-(p/p.shift(3)-1); trend=p/p.shift(60)-1; vol=r.rolling(20,min_periods=15).std()
raw=rev/(vol*np.sqrt(3)+1e-8)*((trend>0).astype(float)*.75+.25)
f=raw.rank(axis=1,pct=True).rolling(3,min_periods=2).mean().shift(1)
for h in [1,5,10,20]:
 z=[];ns=[];fr=p.shift(-h)/p-1
 for dt in p.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=pd.Series(z);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('coverage',round((f.notna().sum(axis=1)/15).mean(),4),'turnover',round(f.diff().abs().mean().mean(),6))
for label,lo in [('2020-25','2020-01-01'),('2026+','2026-01-01'),('2029+','2029-01-01'),('2030','2030-01-01')]:
 z=[];fr=p.shift(-1)/p-1
 for dt in p.index[p.index>=lo]:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=pd.Series(z);print(label,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
f.to_csv('scripts/miner_2_20301017_conditional_reversal_signal.csv')
