import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; cutoff=pd.Timestamp('2026-08-26')
px=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cutoff]; r=px.pct_change()
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].pct_change().reindex(r.index); beta=pd.DataFrame({s:r[s].rolling(60,min_periods=45).cov(d)/d.rolling(60,min_periods=45).var() for s in U}); shock=d.rolling(5,min_periods=5).sum(); f=-beta.mul(shock,axis=0); out=[]
for dt in r.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for h in [5,10]:
 q=[]
 for dt in r.index:
  y=r.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt]; z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q);print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=a[(a.index.year>=lo)&(a.index.year<=hi)].ic; print('regime',lo,hi,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
