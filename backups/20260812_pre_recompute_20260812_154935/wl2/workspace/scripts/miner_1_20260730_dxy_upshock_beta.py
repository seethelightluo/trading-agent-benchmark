import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
def load(p):
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close']; return d.loc[:cut]
dxy=load('../persistent/index_data/DXY.csv').rename('m'); dm=dxy.pct_change(); R={a:load('../persistent/stock_data/'+a+'.csv').pct_change() for a in U}
# Candidate: downside DXY-shock beta, covariance estimated only on days DXY rises; low beta is defensive.
F={}; Y={}
for a in U:
 q=pd.concat([R[a].rename('r'),dm],axis=1,join='inner').dropna(); shock=q.m>0
 # retain aligned observations, zero-weight non-shock via masked rolling moments
 x=q.r.where(shock); m=q.m.where(shock)
 n=x.rolling(90,min_periods=25).count(); mx=x.rolling(90,min_periods=25).mean(); mm=m.rolling(90,min_periods=25).mean()
 cov=(x*m).rolling(90,min_periods=25).mean()-mx*mm; var=m.pow(2).rolling(90,min_periods=25).mean()-mm.pow(2)
 F[a]=(-cov/var).rename(a); Y[a]=R[a].shift(-1).rename(a)
F=pd.concat(F,axis=1); Y=pd.concat(Y,axis=1)
vals=[]; ns=[]; ds=[]
for dt in F.index.intersection(Y.index):
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
a=np.asarray(vals); print('dates',len(a),'range',min(ds),max(ds),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for y in range(2020,2027):
 v=a[[d.year==y for d in ds]]; print(y,len(v),round(v.mean(),5) if len(v) else None,round(v.mean()/v.std(ddof=1),4) if len(v)>1 else None)
for h in [5,10]:
 v=[]
 for dt in F.index:
  if dt not in Y.index: continue
  z=pd.concat([F.loc[dt],Y.shift(-(h-1)).loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.asarray(v);print('horizon',h,'dates',len(v),'IC %.6f ICIR %.6f'%(v.mean(),v.mean()/v.std(ddof=1)))
