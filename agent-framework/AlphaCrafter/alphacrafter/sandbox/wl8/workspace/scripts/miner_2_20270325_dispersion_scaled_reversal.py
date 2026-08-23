import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-03-25'); q={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();q[a]=d.close.loc[:cut]
P=pd.concat(q,axis=1).sort_index(); R=P.pct_change(); e=R.sub(R.median(axis=1),axis=0); disp=e.std(axis=1).rolling(20,min_periods=10).median().shift(1)
f=-(e.rolling(3,min_periods=3).sum().shift(1)).div(disp,axis=0)
for h in [1,5,10]:
 fr=P.pct_change(h).shift(-h); vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x): vals.append(x);ns.append(len(z))
 x=np.array(vals);print('h',h,'dates',len(x),'avg_names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6),'hit',round(np.mean(x>0),4))
rr=f.rank(pct=True,axis=1);t=[]
for i in range(1,len(rr)):
 z=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover',round(np.mean(t),6))
