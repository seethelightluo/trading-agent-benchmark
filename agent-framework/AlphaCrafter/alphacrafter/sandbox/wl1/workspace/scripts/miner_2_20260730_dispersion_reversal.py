import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
end=pd.Timestamp('2026-07-15')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
# Cross-asset dispersion-conditioned reversal: contrarian 5d move, amplified when
# contemporaneous 1d cross-sectional dispersion is unusually high (lagged one day).
allr=[]
for s in assets:
 d=load(s).loc[:end]; allr.append(d.close.pct_change().rename(s))
r=pd.concat(allr,axis=1)
disp=r.where(r.notna().sum(axis=1)>=8).std(axis=1)
z=(disp-disp.rolling(60,min_periods=30).mean())/disp.rolling(60,min_periods=30).std()
rows=[]
for s in assets:
 d=load(s).loc[:end]; mom=d.close.pct_change(5); f=(-mom*z.shift(1)).rename('f'); fr=d.close.shift(-10)/d.close-1
 q=pd.concat([f,fr.rename('r')],axis=1).dropna()
 rows += [(dt,float(a),float(b)) for dt,a,b in zip(q.index,q.f,q.r)]
x=pd.DataFrame(rows,columns=['date','f','r']); vals=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
  vals.append(spearmanr(g.f,g.r).statistic); ns.append(len(g))
a=np.array(vals); print('factor dispersion_conditioned_reversal_10d | dates',len(a),'avg_n',np.mean(ns),'coverage',len(x)/(15*len(a)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover_proxy',np.nan)
for h in [1,5,10]:
 rows=[]
 for s in assets:
  d=load(s).loc[:end]; f=-d.close.pct_change(5)*z.shift(1); fr=d.close.shift(-h)/d.close-1; q=pd.concat([f.rename('f'),fr.rename('r')],axis=1).dropna(); rows += [(dt,a,b) for dt,a,b in zip(q.index,q.f,q.r)]
 q=pd.DataFrame(rows,columns=['date','f','r']); v=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:v.append(spearmanr(g.f,g.r).statistic)
 v=np.array(v); print('h',h,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
for year,g in x.assign(year=x.date.dt.year).groupby('year'):
 v=[]
 for dt,gg in g.groupby('date'):
  if len(gg)>=8 and gg.f.nunique()>1 and gg.r.nunique()>1:v.append(spearmanr(gg.f,gg.r).statistic)
 print('year',year,'IC',np.mean(v),'n',len(v))
