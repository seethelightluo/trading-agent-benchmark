import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv';d=pd.read_csv(p);d['date']=pd.to_datetime(d['date']);return d[d.date<=END].sort_values('date').drop_duplicates('date').set_index('date')['close'].astype(float)
M=load('DXY',1).pct_change();R={s:load(s).pct_change() for s in U}
def factor(a,m,L=60):
 z=pd.concat([a.rename('a'),m.rename('m')],axis=1,join='inner').dropna();x=z.a.values;y=z.m.values;o=np.full(len(z),np.nan)
 for i in range(44,len(z)):
  lo=max(0,i-L+1);v=np.var(y[lo:i+1],ddof=1)
  if v>1e-16:o[i]=-np.cov(x[lo:i+1],y[lo:i+1],ddof=1)[0,1]/v
 return pd.Series(o,index=z.index)
for L in [20,40,60,90,120]:
 rows=[]
 for s in U:
  z=pd.concat([R[s].rename('a'),M.rename('m')],axis=1,join='inner').dropna();f=factor(R[s],M,L)
  for dt in z.index[:-1]:
   if np.isfinite(f.get(dt,np.nan)):rows.append((dt,s,f[dt],z.a.shift(-1).get(dt)))
 d=pd.DataFrame(rows,columns=['date','s','f','y']).dropna();ics=[]
 for dt,g in d.groupby('date'):
  if len(g)>=8:ics.append(spearmanr(g.f,g.y).statistic)
 a=np.array(ics);print('L',L,'dates',len(a),'avgN',round(d.groupby('date').size().mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
