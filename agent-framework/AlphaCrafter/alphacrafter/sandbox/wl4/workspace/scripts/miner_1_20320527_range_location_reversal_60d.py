import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U if os.path.exists(f'../persistent/stock_data/{s}.csv')}
close=pd.DataFrame({s:d.close for s,d in D.items()}); high=pd.DataFrame({s:d.high for s,d in D.items()}); low=pd.DataFrame({s:d.low for s,d in D.items()})
hh=high.rolling(60,min_periods=40).max(); ll=low.rolling(60,min_periods=40).min(); f=.5-(close-ll)/(hh-ll).replace(0,np.nan)
def calc(ret):
 out=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],ret.loc[dt]],axis=1).dropna()
  if len(a)>=8: out.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 return pd.Series(out)
z=calc(close.shift(-10)/close-1)
print('dates',len(z),'avg_n',len(U),'coverage',f.notna().sum().sum()/(f.shape[0]*len(U)))
print('H10 IC ICIR hit',z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
for a,b in [('2020','2023'),('2024','2027'),('2028','2032')]:
 q=calc((close.shift(-10)/close-1)).loc[a:b]; print(a+'-'+b,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
for h in [5,20]:
 q=calc(close.shift(-h)/close-1); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
