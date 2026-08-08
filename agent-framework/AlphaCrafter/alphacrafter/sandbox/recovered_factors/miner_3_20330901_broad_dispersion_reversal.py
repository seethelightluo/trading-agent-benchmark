import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A={f.split('/')[-1].split('.')[0]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
c=pd.DataFrame({a:d.close for a,d in A.items()}).sort_index(); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Continuous, broader dispersion-conditioned reversal: gate is 50th percentile, with a smooth intensity.
disp=r.std(axis=1).rolling(5,min_periods=4).mean(); q=disp.rolling(252,min_periods=100).quantile(.50)
intensity=(disp/q).clip(.5,1.5)
ret5=r.rolling(5,min_periods=4).sum(); peer=ret5.median(axis=1)
s=(-(ret5.sub(peer,axis=0))).div(vol).mul(intensity,axis=0).shift(1)
print('UNIVERSE',len(A),'range',c.index.min().date(),c.index.max().date())
for h in [1,5,10,20]:
 f=c.shift(-h)/c-1; vals=[]; ns=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); print('H',h,'dates',len(x),'meanN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('coverage',round(s.notna().sum().sum()/(s.shape[0]*s.shape[1]),4),'gatefreq',round((disp>q).mean(),4),'turnover10',round(s.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 x=[]
 for dt in s.loc[lo:hi].index:
  z=pd.concat([s.loc[dt],(c.shift(-20)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x); print('REGIME',lo,hi,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
# no persistence: exact library vectors are not serialized, so audit is intentionally explicit
print('LIBRARY_AUDIT exact_max_abs=UNAVAILABLE admitted_factors=26 admission=FAIL')
