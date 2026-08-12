import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(p):
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d[d.date<='2026-07-15'].set_index('date')
m=L('../persistent/index_data/DXY.csv').close.pct_change(); rows=[]
for s in U:
 d=L('../persistent/stock_data/'+s+'.csv');r=d.close.pct_change();z=pd.concat([r,m],axis=1,join='inner');z.columns=['r','m']; b=z.r.rolling(40,min_periods=30).cov(z.m)/z.m.rolling(40,min_periods=30).var(); resid=r-b*m; f=resid.rolling(20,min_periods=15).sum(); y=r.shift(-1);q=pd.concat([f,y],axis=1);q.columns=['f','y'];q['date']=q.index;rows.append(q.reset_index(drop=True))
a=pd.concat(rows,ignore_index=True).dropna();o=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:o.append(spearmanr(g.f,g.y).statistic)
o=pd.Series(o).dropna();print('dates',len(o),'avg names',a.groupby('date').size().mean(),'IC',o.mean(),'ICIR',o.mean()/o.std(),'hit',(o>0).mean(),'std',o.std())
for h in [5,10]:
 rows=[]
 for s in U:
  d=L('../persistent/stock_data/'+s+'.csv');r=d.close.pct_change();z=pd.concat([r,m],axis=1,join='inner');z.columns=['r','m'];b=z.r.rolling(40,min_periods=30).cov(z.m)/z.m.rolling(40,min_periods=30).var();f=(r-b*m).rolling(20,min_periods=15).sum();y=d.close.pct_change(h).shift(-h);q=pd.concat([f,y],axis=1);q.columns=['f','y'];q['date']=q.index;rows.append(q.reset_index(drop=True))
 b=pd.concat(rows,ignore_index=True).dropna();oo=[]
 for _,g in b.groupby('date'):
  if len(g)>=8:oo.append(spearmanr(g.f,g.y).statistic)
 oo=pd.Series(oo).dropna();print(h,oo.mean(),oo.mean()/oo.std(),len(oo))
for a1,a2 in [(2020,2022),(2023,2024),(2025,2026)]:
 # reconstruct date list quickly omitted
 pass
