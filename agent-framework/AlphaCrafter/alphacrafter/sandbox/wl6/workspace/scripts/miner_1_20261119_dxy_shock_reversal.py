import pandas as pd,numpy as np,os,warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'; cut=pd.Timestamp('2026-11-18')
d=pd.read_csv('../persistent/index_data/DXY.csv'); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[:cut]; rr=d.close.pct_change(); q=rr.rolling(60,min_periods=40).quantile(.75).shift(1); gate=((rr>q).astype(float)*(rr.clip(lower=0)/q.replace(0,np.nan)).clip(upper=3)).shift(1)
rows=[]
for a in A:
 p=f'{B}/{a}.csv'
 if not os.path.exists(p):continue
 x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x.sort_values('date').set_index('date').loc[:cut]; r=x.close.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1); f=(-r/vol)*gate.reindex(x.index); rows.append(pd.DataFrame({'f':f,'r':x.close.shift(-1)/x.close-1}).assign(asset=a).reset_index())
z=pd.concat(rows).dropna(subset=['f','r']); o=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
  q=spearmanr(g.f,g.r).statistic
  if np.isfinite(q):o.append((dt,q,len(g)))
z=pd.DataFrame(o,columns=['date','ic','n']); print('dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4),'period',z.date.min().date(),z.date.max().date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-18')]:
 q=z[(z.date>=lo)&(z.date<=hi)];print('regime',lo,hi,'dates',len(q),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5))
