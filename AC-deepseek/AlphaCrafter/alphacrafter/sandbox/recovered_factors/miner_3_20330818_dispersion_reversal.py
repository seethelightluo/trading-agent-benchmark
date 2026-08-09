import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A={Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
c=pd.DataFrame({a:d.close for a,d in A.items()}).sort_index()
r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Candidate: idiosyncratic 5d reversal, activated only on high cross-sectional dispersion.
# Dispersion is observable at t-1; signal is lagged one completed day.
csdisp=r.rolling(5,min_periods=4).std(axis=1)
q=csdisp.rolling(252,min_periods=100).quantile(.70)
gate=(csdisp>q).astype(float)
csmed=r.rolling(5,min_periods=4).median(axis=1)
s=(-(r.rolling(5,min_periods=4).sum().sub(r.rolling(5,min_periods=4).sum(axis=1),axis=0) if False else (r.rolling(5,min_periods=4).sum().sub(r.rolling(5,min_periods=4).median(axis=1),axis=0))) / vol).mul(gate,axis=0).shift(1)
for h in [1,5,10,20]:
 f=c.shift(-h)/c-1; vals=[]; ns=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); print('H',h,'dates',len(x),'meanN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('assets',len(A),'range',c.index.min(),c.index.max(),'coverage',round(s.notna().sum().sum()/(s.shape[0]*s.shape[1]),4),'gatefreq',round((gate>0).mean().mean(),4))
print('turnover10',round(s.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 x=[]
 for dt in s.loc[lo:hi].index:
  z=pd.concat([s.loc[dt],(c.shift(-5)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x); print('regime',lo,hi,'dates',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
