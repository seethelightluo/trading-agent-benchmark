import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(p):
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d[d.date<='2026-07-15'].set_index('date')
m=L('../persistent/index_data/DXY.csv').close.pct_change()
for lb in [5,10,20,40]:
 rows=[]
 for s in U:
  d=L('../persistent/stock_data/'+s+'.csv'); r=d.close.pct_change(); z=pd.concat([r,m],axis=1,join='inner');z.columns=['r','m']
  beta=z.r.rolling(40,min_periods=30).cov(z.m)/z.m.rolling(40,min_periods=30).var()
  e=r-beta*m; f=e.rolling(lb,min_periods=max(3,lb//2)).sum()/e.rolling(20,min_periods=10).std()
  q=pd.concat([f,r.shift(-1)],axis=1);q.columns=['f','y'];q['date']=q.index;rows.append(q.reset_index(drop=True))
 a=pd.concat(rows,ignore_index=True).dropna(); obs=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 o=pd.DataFrame(obs,columns=['date','ic','n']).dropna(); print('lb',lb,'dates',len(o),'avgN',o.n.mean(),'coverage',len(o)/a.date.nunique(),'IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean(),'turn',a.groupby('date').f.rank(pct=True).groupby(a.s if 's' in a else a.date).size().mean() if False else 'NA')
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  x=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic;print(' regime',lo,hi,x.mean(),x.mean()/x.std(),len(x))
