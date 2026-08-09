import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); d=d[d.date<='2026-07-15']; return d.set_index('date')
xs={s:load('../persistent/stock_data/'+s+'.csv') for s in U}; macro=load('../persistent/index_data/DXY.csv')
# DXY-beta defensive signal: negative rolling covariance / variance, 60 observations, aligned by dates
mr=macro.close.pct_change()
rows=[]; ranks=[]
for s,d in xs.items():
 r=d.close.pct_change(); z=pd.concat([r,mr],axis=1,join='inner'); z.columns=['r','m']
 cov=z.r.rolling(60,min_periods=45).cov(z.m); var=z.m.rolling(60,min_periods=45).var()
 f=-(cov/var); fr=r.shift(-1)
 q=pd.concat([f,fr],axis=1); q.columns=['f','y']; q['s']=s; rows.append(q.reset_index())
a=pd.concat(rows,ignore_index=True).dropna(); obs=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  ic=spearmanr(g.f,g.y).statistic
  obs.append((dt,ic,len(g),g.f.rank().corr(g.y.rank())))
o=pd.DataFrame(obs,columns=['date','ic','n','x']); o=o.dropna();
print('dates',len(o),'avg_n',o.n.mean(),'coverage',len(o)/a.date.nunique(),'IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean(),'std',o.ic.std())
for h in [5,10]:
 rr=[]
 for s,d in xs.items():
  r=d.close.pct_change(); z=pd.concat([r,mr],axis=1,join='inner'); cov=z.iloc[:,0].rolling(60,min_periods=45).cov(z.iloc[:,1]); var=z.iloc[:,1].rolling(60,min_periods=45).var(); f=-(cov/var); y=d.close.pct_change(h).shift(-h); q=pd.concat([f,y],axis=1).dropna(); q.columns=['f','y'];q['date']=q.index;rr.append(q)
 b=pd.concat(rr).reset_index(drop=True); oo=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8: oo.append(spearmanr(g.f,g.y).statistic)
 oo=pd.Series(oo).dropna();print(h,'d IC',oo.mean(),'ICIR',oo.mean()/oo.std(),'obs',len(oo))
# rank turnover
rank=a.assign(rk=a.groupby('date').f.rank(pct=True)).sort_values(['s','date']); print('turnover',rank.groupby('s').rk.diff().abs().mean().mean())
# regime
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 x=o[(o.date.dt.year>=int(lo))&(o.date.dt.year<=int(hi))].ic; print(lo+'-'+hi,x.mean(),x.mean()/x.std(),len(x))
# correlations with persisted factor values pooled
for fpath in glob.glob('factors/*.json'):
 pass
