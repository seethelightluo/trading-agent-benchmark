import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-02-23')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fwd=px.shift(-1)/px-1
# relative strength: lagged 10d asset return minus contemporaneous cross-sectional median
rel=r.rolling(10).sum().sub(r.rolling(10).sum().median(axis=1),axis=0).shift(1)
vals=[]; ds=[]; ns=[]
for dt in px.index:
 g=pd.DataFrame({'s':rel.loc[dt],'f':fwd.loc[dt]}).dropna()
 if len(g)>=8:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q);ds.append(dt);ns.append(len(g))
a=np.array(vals);print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(rel.notna().sum().sum()/rel.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lab,fn in [('2026',lambda d:d.year==2026),('2027+',lambda d:d.year>=2027),('recent180',lambda d:d>=END-pd.Timedelta(days=180))]:
 z=a[[i for i,d in enumerate(ds) if fn(d)]];print(lab,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),len(z))
print('turnover',round(float(rel.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
