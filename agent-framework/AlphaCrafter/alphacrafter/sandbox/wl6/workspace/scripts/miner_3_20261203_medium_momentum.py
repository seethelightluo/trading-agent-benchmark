import pandas as pd,numpy as np,os,warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'; cut=pd.Timestamp('2026-12-02'); P={}
for a in A:
 p=f'{B}/{a}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
P=pd.DataFrame(P); R=P.pct_change(); vol=R.rolling(20,min_periods=15).std().shift(1); fac=(P.shift(1)/P.shift(61)-1).div(vol)
def calc(h):
 rows=[]
 for a in P: rows.append(pd.DataFrame({'date':P.index,'f':fac[a].values,'r':(P[a].shift(-h)/P[a]-1).values}))
 d=pd.concat(rows,ignore_index=True).dropna();o=[]
 for dt,g in d.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   x=spearmanr(g.f,g.r).statistic
   if np.isfinite(x):o.append((dt,x,len(g)))
 return pd.DataFrame(o,columns=['date','ic','n'])
for h in [1,5,10]:
 z=calc(h);m=z.ic.mean();print(h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'IC',round(m,5),'ICIR',round(m/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1);print('period',z.date.min().date(),z.date.max().date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-02')]:
 q=z[(z.date>=lo)&(z.date<=hi)];print('regime',lo,hi,'dates',len(q),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5))
