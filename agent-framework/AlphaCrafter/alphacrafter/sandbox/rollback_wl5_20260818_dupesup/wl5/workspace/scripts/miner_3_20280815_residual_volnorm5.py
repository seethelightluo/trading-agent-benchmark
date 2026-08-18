import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-08-14'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill(); R=P.pct_change(); bench=R.mean(axis=1)
beta=R.rolling(60,min_periods=30).cov(bench).div(bench.rolling(60,min_periods=30).var(),axis=0)
res=P.pct_change(5)-beta.mul(bench.rolling(5).sum(),axis=0)
idvol=(R-beta.mul(bench,axis=0)).rolling(20,min_periods=10).std()
f=(-res/idvol).replace([np.inf,-np.inf],np.nan); y=P.shift(-10)/P-1
A=[];N=[];D=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);N.append(len(z));D.append(dt)
a=np.array(A); print('ALL dates',len(a),'avgN',np.mean(N),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-08-14')]:
 q=(np.array(D)>=pd.Timestamp(lo))&(np.array(D)<=pd.Timestamp(hi));c=a[q];print('REG',lo,len(c),c.mean(),c.mean()/c.std(ddof=1),np.mean(c>0))
r=f.rank(axis=1,pct=True);print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',(r-r.shift()).abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_3_20280815_residual_volnorm5_signal.csv')
for h in [1,3,5,10,20]:
 yy=P.shift(-h)/P-1;q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('DECAY',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
print('period',P.index.min().date(),P.index.max().date())
