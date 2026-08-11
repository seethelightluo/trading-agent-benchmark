import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(p):
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close; return d.loc[:cut]
# Equal-weight market residual momentum: remove rolling beta to the contemporaneous cross-asset basket, then rank residual trend.
P=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in U},axis=1).sort_index(); R=P.pct_change(); M=R.mean(axis=1)
F={};Y={}
for a in U:
 q=pd.concat([R[a].rename('r'),M.rename('m')],axis=1).dropna()
 cov=q.r.rolling(60,min_periods=45).cov(q.m); var=q.m.rolling(60,min_periods=45).var()
 resid=(q.r-(cov/var)*q.m).rolling(20,min_periods=15).sum()
 F[a]=resid.rename(a);Y[a]=R[a].shift(-1).rename(a)
F=pd.concat(F,axis=1);Y=pd.concat(Y,axis=1)
ics=[];ds=[];ns=[]
for dt in F.index.intersection(Y.index):
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
a=np.asarray(ics);print('dates',len(a),'range',min(ds),max(ds),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for y in range(2020,2027):
 v=a[[d.year==y for d in ds]];print(y,len(v),round(v.mean(),5),round(v.mean()/v.std(ddof=1),4))
for h in [5,10]:
 v=[]
 for dt in F.index:
  if dt not in Y.index:continue
  z=pd.concat([F.loc[dt],Y.shift(-(h-1)).loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.asarray(v);print('horizon',h,'dates',len(v),'IC %.6f ICIR %.6f'%(v.mean(),v.mean()/v.std(ddof=1)))
