import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
A=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
def r(f):
 d=pd.read_csv(f,parse_dates=['date']).set_index('date');return d.close.pct_change()
R=pd.concat({a:r('../persistent/stock_data/'+a+'.csv') for a in A},axis=1)
V=r('../persistent/index_data/VIX.csv'); shock=V.rolling(60,min_periods=30).quantile(.75)
# resilience: mean return on prior high-VIX-return days minus unconditional mean, scaled by downside deviation
hi=V>shock
for w in [40,60,90]:
 # trailing conditional average; requires aligned masks and completed dates
 num=R.where(hi, np.nan).rolling(w,min_periods=12).mean(); base=R.rolling(w,min_periods=30).mean(); downside=R.where(R<0).rolling(w,min_periods=12).std()
 fac=(num-base).div(downside.replace(0,np.nan))
 fwd=R.shift(-1); out=[]; dates=[]; ns=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna(); y=fwd.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1:out.append(spearmanr(x,y).statistic);dates.append(dt);ns.append(len(x))
 s=pd.Series(out,index=pd.DatetimeIndex(dates));print('W',w,'dates',len(s),'avgN',np.mean(ns),'IC %.5f ICIR %.5f hit %.3f'%(s.mean(),s.mean()/s.std(),(s>0).mean()),'coverage',fac.notna().sum(axis=1).mean()/15)
 for h in [5,10]:
  out=[]
  for dt in fac.index:
   x=fac.loc[dt].dropna();y=R.shift(-h).loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
   if len(x)>=8 and x.nunique()>1 and y.nunique()>1:out.append(spearmanr(x,y).statistic)
  z=pd.Series(out);print(' h',h,len(z),'IC %.5f IR %.5f'%(z.mean(),z.mean()/z.std()))
 print('regimes',[(p,round(s[s.index.year==p].mean(),4),len(s[s.index.year==p])) for p in range(2020,2027)])
