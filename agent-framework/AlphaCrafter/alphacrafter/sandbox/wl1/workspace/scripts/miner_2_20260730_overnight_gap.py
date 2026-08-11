import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-07-15';F={};C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]; C[s]=d.close
 gap=d.open/d.close.shift(1)-1
 # overnight gap continuation, volatility normalized, 5-session average
 F[s]=(gap.rolling(5,min_periods=3).mean()/d.close.pct_change().rolling(20,min_periods=10).std()).rename(s)
p=pd.DataFrame(C).sort_index();fac=pd.DataFrame(F).reindex(p.index)
def run(y):
 a=[];n=[];ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);n.append(len(z));ds.append(pd.Timestamp(dt))
 return np.array(a),np.array(n),pd.DatetimeIndex(ds)
a,n,ds=run(p.shift(-1)/p-1);print('candidate overnight_gap_5d','dates',len(a),'avgN',round(n.mean(),2),'coverage',round(n.mean()/15,4),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4),'turnover',round(fac.rank(pct=True).diff().abs().mean().mean(),5))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(ds>=lo)&(ds<=hi)];print('regime',lo,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5))
for h in [5,10]:
 q,_,_=run(p.shift(-h)/p-1);print('horizon',h,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5))
for name,x in [('rev5',-(p/p.shift(5)-1)),('mom20',p/p.shift(20)-1)]:print('corr',name,round(fac.stack().corr(x.stack()),5))
