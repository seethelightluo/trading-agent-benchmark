import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2030-09-04');b=Path('../persistent/stock_data')
D={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut] for s in U};idx=sorted(set().union(*[set(x.index) for x in D.values()]));p=pd.DataFrame({s:x.reindex(idx) for s,x in D.items()}).ffill();r=p.pct_change()
# Downside-adjusted medium-term trend, cross-sectional percentile, then smooth percentile.
down=r.clip(upper=0).pow(2).rolling(40,min_periods=25).mean().pow(.5); trend=p/p.shift(30)-1
raw=trend/(down*np.sqrt(30)+1e-8)
rank=raw.rank(axis=1,pct=True);f=rank.rolling(10,min_periods=5).mean().shift(1)
fr=p.shift(-1)/p-1
for h in [1,5,10,20]:
 z=[];ns=[]
 for dt in p.index:
  q=pd.concat([f.loc[dt],(p.shift(-h)/p-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=pd.Series(z);print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('coverage',(f.notna().sum(axis=1)/15).mean(),'rank turnover',f.diff().abs().mean().mean())
for label,lo in [('2020-25','2020-01-01'),('2026+','2026-01-01'),('2029+','2029-01-01'),('2030','2030-01-01')]:
 z=[]
 for dt in p.index[p.index>=lo]:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=pd.Series(z);print(label,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_2_20300905_downside_adjusted_trend_signal.csv')
