import pandas as pd, numpy as np, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2026-12-02')
px={}
for a in assets:
 p=f'{base}/{a}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); px[a]=d.sort_values('date').set_index('date').close.loc[:cutoff]
P=pd.DataFrame(px); R=P.pct_change(); disp=R.std(axis=1); threshold=disp.rolling(60,min_periods=40).quantile(.70).shift(1); active=(disp>threshold).shift(1).fillna(False)
lag=R.shift(1); vol=R.rolling(20,min_periods=15).std().shift(1); fac=-(lag.sub(lag.mean(axis=1),axis=0)).div(vol).mul(active.astype(float),axis=0)
def calc(h):
 rows=[]
 for a in P.columns: rows.append(pd.DataFrame({'date':P.index,'f':fac[a].values,'r':(P[a].shift(-h)/P[a]-1).values}))
 dd=pd.concat(rows,ignore_index=True).dropna(); obs=[]
 for dt,g in dd.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   x=spearmanr(g.f,g.r).statistic
   if np.isfinite(x): obs.append((dt,x,len(g)))
 return pd.DataFrame(obs,columns=['date','ic','n'])
for h in [1,5,10]:
 z=calc(h); m=z.ic.mean(); print(h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'IC',round(m,5),'ICIR',round(m/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1); print('period',z.date.min().date(),z.date.max().date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-02')]:
 q=z[(z.date>=lo)&(z.date<=hi)]; print('regime',lo,hi,'dates',len(q),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5))
