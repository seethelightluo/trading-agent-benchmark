import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d[d.date<=END].sort_values('date').drop_duplicates('date')
 return d.set_index('date')['close'].astype(float)
R={s:load(s).pct_change() for s in U}; M=load('DXY',True).pct_change()
def sig(a,m,L=60,minp=45):
 z=pd.concat([a.rename('a'),m.rename('m')],axis=1,join='inner').dropna(); out=[]
 for i,dt in enumerate(z.index):
  lo=max(0,i-L+1); x=z.a.iloc[lo:i+1]; y=z.m.iloc[lo:i+1]
  if len(x)>=minp and y.var()>1e-16: out.append((dt,-x.cov(y)/y.var()))
 return pd.Series(dict(out))
rows=[]
for s in U:
 f=sig(R[s],M); y=R[s].shift(-1)
 for dt,v in f.items():
  if dt in y.index and np.isfinite(y[dt]): rows.append((dt,s,v,y[dt]))
d=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[]
for dt,g in d.groupby('date'):
 if len(g)>=8: ics.append(spearmanr(g.f,g.y).statistic)
a=np.asarray(ics); print('dates',len(a),'instruments',d.s.nunique(),'avgN',d.groupby('date').size().mean(),'coverage',len(d)/(len(R)*len(set(d.date))))
print('daily IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for h in [5,10]:
 z=[]
 for s in U:
  f=sig(R[s],M); y=R[s].rolling(h).sum().shift(-h+1)
  for dt,v in f.items():
   if dt in y.index and np.isfinite(y[dt]): z.append((dt,s,v,y[dt]))
 q=pd.DataFrame(z,columns=['date','s','f','y']); aa=np.array([spearmanr(g.f,g.y).statistic for _,g in q.groupby('date') if len(g)>=8]); print('h',h,'dates',len(aa),'IC',aa.mean(),'ICIR',aa.mean()/aa.std(ddof=1))
# rank turnover
p=d.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean(),'first',d.date.min(),'last',d.date.max())
# yearly means
print(d.assign(ic=np.nan).groupby(d.date.dt.year).size().to_dict())
