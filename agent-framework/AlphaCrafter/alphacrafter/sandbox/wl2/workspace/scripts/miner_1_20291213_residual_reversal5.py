import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date') for s in U}
cl=pd.DataFrame({s:d.close for s,d in D.items()}); r=cl.pct_change(); v=pd.DataFrame({s:d.volume for s,d in D.items()})
m=r.rolling(60,min_periods=40).std(); beta=r.rolling(60,min_periods=40).cov(r.mean(axis=1)).div(r.mean(axis=1).rolling(60,min_periods=40).var(),axis=0)
res=r-beta*r.mean(axis=1).values[:,None]
# 5d residual reversal, amplified by unusual volume, volatility normalized
base=-res.rolling(5,min_periods=5).sum()/m
activity=(v/v.rolling(20,min_periods=10).mean()).clip(0.5,2)-1
f=base*(1+0.35*activity)
# avoid future: rows date are signal at close, forward returns
for h in [1,5,10]:
 q=[]; ns=[]
 for dt in f.index:
  x=f.loc[dt]; y=r.shift(-h).loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
 q=pd.Series(q).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean(),'n',len(q))
print('avg instruments',np.mean(ns),'coverage',np.mean(ns)/15)
# date series h1 for subregimes
q=[]
for dt in f.index:
 x=f.loc[dt];y=r.shift(-1).loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:q.append((dt,spearmanr(x[ok],y[ok]).statistic))
a=pd.DataFrame(q,columns=['date','ic']).set_index('date')
for s in ['2026-07-16','2028-01-01','2029-01-01','2029-07-01']:
 z=a.loc[s:].ic;print(s,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20291213_residual_reversal5_signal.csv',index=False)
