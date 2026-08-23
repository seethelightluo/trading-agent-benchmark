import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change()
# short-term residual reversal: reverse 3-day move relative to cross-asset median, risk normalized
raw=r.rolling(3,min_periods=3).sum(); resid=raw.sub(raw.median(axis=1),axis=0)
f=-resid/r.rolling(20,min_periods=10).std()
for h in [5,10,20]:
 x=[]
 for d in p.index:
  if d>pd.Timestamp('2030-06-26'): break
  z=pd.concat([f.loc[d],p.shift(-h).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('h',h,'dates',len(x),'ic',x.mean(),'icir',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for lo,hi in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2024-12-31')),(pd.Timestamp('2025-01-01'),pd.Timestamp('2027-12-31')),(pd.Timestamp('2028-01-01'),pd.Timestamp('2030-06-26'))]:
 x=[]
 for d in p.index[(p.index>=lo)&(p.index<=hi)]:
  z=pd.concat([f.loc[d],p.shift(-10).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('reg',lo.year,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('coverage',f.loc[:'2030-06-26'].notna().stack().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20300627_residual_reversal_signal.csv')
