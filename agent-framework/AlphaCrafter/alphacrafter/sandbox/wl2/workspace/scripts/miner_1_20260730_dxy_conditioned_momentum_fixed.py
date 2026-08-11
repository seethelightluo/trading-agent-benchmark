import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(p):
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close; return d.loc[:cut]
# Explicitly compute macro regime on each asset's own calendar; do not broadcast through a union calendar.
D=load('../persistent/index_data/DXY.csv').pct_change()
F={}; Y={}
for a in U:
 p=load('../persistent/stock_data/'+a+'.csv'); r=p.pct_change()
 q=pd.concat([r.rename('r'),D.rename('d')],axis=1,join='inner').dropna()
 reg=-q.d.rolling(20,min_periods=15).sum()
 mom=q.r.rolling(20,min_periods=15).sum()
 F[a]=(mom*reg).rename(a)
 Y[a]=r.shift(-1).rename(a)
F=pd.concat(F,axis=1); Y=pd.concat(Y,axis=1)
ics=[];ds=[];ns=[]
for dt in F.index.intersection(Y.index):
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
a=np.asarray(ics)
print('dates',len(a),'range',min(ds) if ds else None,max(ds) if ds else None,'avgN',round(np.mean(ns),2) if ns else None)
print('IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()) if len(a)>1 else 'insufficient')
print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for y in range(2020,2027):
 v=a[[d.year==y for d in ds]]; print(y,len(v),round(v.mean(),5) if len(v) else None,round(v.mean()/v.std(ddof=1),4) if len(v)>1 else None)
for h in [5,10]:
 vals=[]
 for dt in F.index:
  if dt not in Y.index: continue
  z=pd.concat([F.loc[dt],Y.shift(-(h-1)).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.asarray(vals); print('horizon',h,'dates',len(v),'IC %.6f ICIR %.6f'%(v.mean(),v.mean()/v.std(ddof=1)))
# This is exploratory only; no persistence unless both daily gates pass.
