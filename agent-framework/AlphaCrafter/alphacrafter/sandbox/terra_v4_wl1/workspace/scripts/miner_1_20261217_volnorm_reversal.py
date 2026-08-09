import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index();
# Volatility-adjusted 20-day reversal: negative recent return scaled by realized risk.
ret=R.rolling(20,min_periods=15).sum(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20); F=-ret/vol
rows=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),R.shift(-1).loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=a.ic
print('dates',len(q),'range',a.index.min().date(),a.index.max().date(),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'year',a.groupby(a.index.year).ic.mean().round(4).to_dict())
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for k in [5,10]:
 y=R.shift(-k).rolling(k).sum().shift(-(k-1)); zq=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:zq.append(spearmanr(z.f,z.y).statistic)
 zq=pd.Series(zq);print(k,zq.mean(),zq.mean()/zq.std(ddof=1),len(zq))
