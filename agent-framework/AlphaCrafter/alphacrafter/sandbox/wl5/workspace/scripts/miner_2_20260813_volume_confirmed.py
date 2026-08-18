import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d[d.date<=END].sort_values('date').drop_duplicates('date').set_index('date')
D={s:load(s) for s in U}; rows=[]
for s,x in D.items():
 r=x.close.pct_change(); # completed-day signal, volume surprise confirmation
 vs=np.log(x.volume.replace(0,np.nan)/x.volume.rolling(60,min_periods=30).median())
 f=r.rolling(5).sum()*vs
 y=r.shift(-1)
 for dt in f.index:
  if np.isfinite(f.get(dt,np.nan)) and np.isfinite(y.get(dt,np.nan)): rows.append((dt,s,f[dt],y[dt]))
d=pd.DataFrame(rows,columns=['date','s','f','y']);a=np.array([spearmanr(g.f,g.y).statistic for _,g in d.groupby('date') if len(g)>=8]);print('dates',len(a),'instruments',d.s.nunique(),'avgN',d.groupby('date').size().mean(),'coverage',len(d)/(15*len(set(d.date))));print('daily',a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
for h in [5,10]:
 q=[]
 for s,x in D.items():
  r=x.close.pct_change();vs=np.log(x.volume.replace(0,np.nan)/x.volume.rolling(60,min_periods=30).median());f=r.rolling(5).sum()*vs;y=r.rolling(h).sum().shift(-h+1)
  for dt,v in f.items():
   if np.isfinite(v) and np.isfinite(y.get(dt,np.nan)):q.append((dt,s,v,y[dt]))
 z=pd.DataFrame(q,columns=['date','s','f','y']);aa=np.array([spearmanr(g.f,g.y).statistic for _,g in z.groupby('date') if len(g)>=8]);print('h',h,len(aa),aa.mean(),aa.mean()/aa.std(ddof=1))
print('turnover',d.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True).diff().abs().mean().mean())
