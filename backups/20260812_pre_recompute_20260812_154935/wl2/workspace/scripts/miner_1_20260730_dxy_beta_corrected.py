import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
def close(path):
 d=pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index()
 return d.loc[:cut]
dxy=close('../persistent/index_data/DXY.csv').rename('dxy')
dr=dxy.pct_change()
rets={a:close('../persistent/stock_data/'+a+'.csv').pct_change() for a in U}
fac={}; fwd={}
for a in U:
 # explicit inner date alignment avoids concat calendar artifacts
 q=pd.concat([rets[a].rename('r'),dr],axis=1,join='inner').dropna()
 cov=q.r.rolling(60,min_periods=45).cov(q.dxy); var=q.dxy.rolling(60,min_periods=45).var()
 fac[a]=(-cov/var).rename(a)
 fwd[a]=rets[a].shift(-1).rename(a)
F=pd.concat(fac,axis=1); Y=pd.concat(fwd,axis=1)
ics=[]; counts=[]; dates=[]
for dt in F.index.intersection(Y.index):
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); counts.append(len(z)); dates.append(dt)
a=np.asarray(ics); print('dates',len(a),'range',min(dates),max(dates),'meanN',round(np.mean(counts),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for y in range(2020,2027):
 v=a[[d.year==y for d in dates]]
 print(y,len(v),round(v.mean(),5) if len(v) else None,round(v.mean()/v.std(ddof=1),4) if len(v)>1 else None)
for h in [5,10]:
 vals=[]; ds=[]
 for dt in F.index:
  if dt not in Y.index: continue
  z=pd.concat([F.loc[dt],Y.shift(-(h-1)).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt)
 v=np.asarray(vals); print('horizon',h,'dates',len(v),'IC %.6f ICIR %.6f'%(v.mean(),v.mean()/v.std(ddof=1)))
