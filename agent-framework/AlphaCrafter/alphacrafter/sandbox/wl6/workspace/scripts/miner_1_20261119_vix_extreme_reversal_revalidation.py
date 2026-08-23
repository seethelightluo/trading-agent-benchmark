import pandas as pd, numpy as np, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-11-18'); base='../persistent/stock_data'
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.sort_values('date').set_index('date').loc[:cutoff]
vr=v.close.pct_change(); q=vr.rolling(60,min_periods=40).quantile(.75).shift(1)
shock=((vr>q).astype(float)*(vr.clip(lower=0)/q.replace(0,np.nan)).clip(upper=3)).shift(1)
rows=[]
for a in assets:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[:cutoff]
 ret=d.close.pct_change(); vol=ret.rolling(20,min_periods=15).std().shift(1)
 fac=(-ret/vol)*shock.reindex(d.index)
 z=pd.DataFrame({'f':fac,'r':d.close.shift(-1)/d.close-1,'asset':a}).reset_index(); rows.append(z)
dd=pd.concat(rows,ignore_index=True).dropna(subset=['f','r']); obs=[]
for dt,g in dd.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
  x=spearmanr(g.f,g.r).statistic
  if np.isfinite(x): obs.append((dt,x,len(g)))
z=pd.DataFrame(obs,columns=['date','ic','n']); print('dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4),'period',z.date.min().date(),z.date.max().date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-18')]:
 qz=z[(z.date>=lo)&(z.date<=hi)]; print('regime',lo,hi,'dates',len(qz),'IC',round(qz.ic.mean(),5),'ICIR',round(qz.ic.mean()/qz.ic.std(ddof=1),5))
