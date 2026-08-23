import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change()
# medium-term trend persistence, risk adjusted by recent volatility; fully lagged at decision date
f=p.pct_change(60)/r.rolling(20,min_periods=20).std()
# cross-sectional normalize
f=f.sub(f.median(axis=1),axis=0)
for h in [5,10,20]:
 x=[]
 for d in p.index:
  if d>pd.Timestamp('2030-06-26'): break
  z=pd.concat([f.loc[d],p.shift(-h).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.asarray(x); print(h,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-06-26')]:
 x=[]
 for d in p.index:
  if not(lo<=str(d.date())<=hi):continue
  z=pd.concat([f.loc[d],p.shift(-20).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('reg',lo,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_2_20300627_medium_trend_signal.csv')
print('coverage',f.notna().stack().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
