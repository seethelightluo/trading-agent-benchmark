import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-03-12')
p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); p[a]=d.close[d.index<=end]
x=pd.DataFrame(p); r=x.pct_change(); r5=x.pct_change(5); med=r5.median(axis=1)
# candidate: residual 5d reversal, normalized by downside/idiosyncratic vol and conditioned on market dispersion
res=r5.sub(med,axis=0)
idv=r.sub(r.median(axis=1),axis=0).rolling(30,min_periods=15).std()
down=r.where(r<0).sub(r.where(r<0).median(axis=1),axis=0).rolling(30,min_periods=15).std()
disp=r.std(axis=1).rolling(20,min_periods=10).mean(); dz=(disp-disp.rolling(120,min_periods=40).mean())/disp.rolling(120,min_periods=40).std()
for name,den,cond in [('idvol',idv,1),('downvol',down,1),('idvol_stress',idv,(1+dz.clip(-1,2)) )]:
 s=-res/den.replace(0,np.nan)*cond
 f=x.pct_change(5).shift(-5); ic=[]; ns=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):ic.append(q);ns.append(len(z))
 q=np.array(ic); print(name,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
