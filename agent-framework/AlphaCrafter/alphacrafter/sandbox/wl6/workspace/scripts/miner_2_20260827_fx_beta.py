import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
P=pd.DataFrame(px).sort_index(); R=P.pct_change();
for macro in ['EURUSD','USDJPY']:
 m=pd.read_csv('../persistent/index_data/'+macro+'.csv',parse_dates=['date']).set_index('date')['close']; mr=m.pct_change()
 f=pd.DataFrame(index=R.index,columns=U,dtype=float)
 for s in U: f[s]=-R[s].rolling(60,min_periods=45).cov(mr)/mr.rolling(60,min_periods=45).var()
 out=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append(q);ns.append(len(z))
 a=np.array(out); print(macro,'dates',len(a),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for h in [5,10]:
  q=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  q=np.array(q);print(' h',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'n',len(q))
 for y in range(2020,2027):
  q=np.array([v for dt,v in zip(f.index[-len(out):],out) if dt.year==y])
  if len(q)>1:print(' yr',y,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
