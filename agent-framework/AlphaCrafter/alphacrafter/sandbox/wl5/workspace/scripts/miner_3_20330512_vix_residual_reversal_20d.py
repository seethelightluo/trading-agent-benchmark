import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=150:px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); mr=R['SPX']
beta=R.rolling(60,min_periods=40).cov(mr).div(mr.rolling(60,min_periods=40).var(),axis=0)
res=R.sub(beta.mul(mr,axis=0),axis=0).rolling(20,min_periods=15).sum()
v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].reindex(P.index).ffill()
med=v.rolling(252,min_periods=100).quantile(.6); stress=(v>med).astype(float)
# negative residual return under elevated VIX, retain a small neutral baseline to preserve ranking
f=-res*(0.25+0.75*stress.values[:,None]); fr=R.shift(-10).rolling(10,min_periods=10).sum()
f.to_csv('scripts/miner_3_20330512_vix_residual_reversal_20d_signal.csv'); ics=[];dates=[];ns=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);dates.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
a=np.array(ics);S=pd.DataFrame(ranks,index=dates)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':np.mean(ns),'coverage':np.mean(ns)/15,'IC':a.mean(),'ICIR':a.mean()/a.std(ddof=1)*np.sqrt(252),'hit':np.mean(a>0),'turnover':S.diff().abs().mean().mean()})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-05-11')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))];print(x,len(z),z.mean() if len(z) else None,z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else None)
