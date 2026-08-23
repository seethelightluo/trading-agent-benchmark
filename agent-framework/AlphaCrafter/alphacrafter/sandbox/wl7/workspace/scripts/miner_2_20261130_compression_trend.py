import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.DataFrame(p).sort_index().loc[:'2026-11-30']; r=p.pct_change()
# persistent trend rewarded when short volatility is below long volatility; fully lagged
f=(p.shift(1)/p.shift(41)-1)*(1-r.shift(1).rolling(5,min_periods=5).std()/(r.shift(1).rolling(60,min_periods=40).std()+1e-8))
f=f.replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 y=p.shift(-h)/p-1; a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(a);print('h',h,'n',len(a),'names',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('turnover',np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=np.array([v for d,v in zip(ds,a) if lo<=str(d.year)<=hi]); print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else 0)
