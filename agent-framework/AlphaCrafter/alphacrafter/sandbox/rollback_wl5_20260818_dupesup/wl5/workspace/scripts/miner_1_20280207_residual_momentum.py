import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-06'); b=Path('../persistent/stock_data')
px={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change(); m=R.mean(axis=1)
# causal 20-day momentum, residualized against contemporaneous 60d market beta and normalized by idiosyncratic volatility
mom=P.pct_change(20); beta=R.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
market20=m.rolling(20).sum(); resid=mom-beta.mul(market20,axis=0)
idio=(R-beta.mul(m,axis=0)).rolling(20,min_periods=10).std()*np.sqrt(252)
f=resid/(idio+1e-8); y=P.shift(-10)/P-1
ics=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
a=np.asarray(ics); print('cutoff',end.date(),'dates',len(a),'meanN',round(np.mean(ns),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),4))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-06')]:
 q=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.asarray(q); print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
f.to_csv('scripts/miner_1_20280207_residual_momentum_signal.csv')
