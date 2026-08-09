import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date)
 D[s]=x.set_index('date').sort_index().close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); R=R.loc[:'2026-12-16']
# downside-risk-adjusted short momentum: recent return per downside deviation, all lagged at completed day
ret=R.rolling(15,min_periods=10).sum(); down=R.where(R<0).rolling(30,min_periods=10).std(); F=ret/(down*np.sqrt(15)+1e-8)
rows=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),R.shift(-1).loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('idea downside-adjusted 15d momentum')
for k in [1,5,10]:
 y=R.shift(-k).rolling(k).sum().shift(-(k-1)); q=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 q=pd.Series(q).dropna(); print(k,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q))
print('blocks',a.groupby(pd.cut(a.index.year,[2019,2022,2024,2026])).ic.mean().to_dict(),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
# signal artifact
F.to_csv('scripts/miner_3_20261217_downside_mom_signal.csv')
