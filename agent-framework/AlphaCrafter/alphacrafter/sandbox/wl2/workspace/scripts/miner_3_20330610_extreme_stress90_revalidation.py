import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 z=pd.read_csv('../persistent/stock_data/'+s+'.csv'); z.date=pd.to_datetime(z.date); return z.set_index('date').sort_index()
d={s:ld(s) for s in U}; c=pd.concat({s:z.close for s,z in d.items()},axis=1); R=np.log(c).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); V=vix.set_index('date').close.reindex(c.index).ffill()
res=R.sub(R.mean(axis=1),axis=0); rv=res.rolling(5,min_periods=4).sum().shift(1); vol=R.rolling(20,min_periods=10).std().shift(1)
q=V.shift(1).rolling(252,min_periods=60).quantile(.90); breadth=(R<0).mean(axis=1).shift(1)
gate=((V.shift(1)>q)&(breadth>=.80)); f=(-rv/vol).where(gate, np.nan)
resu=[]
for dt in R.index:
 z=pd.concat([f.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: resu.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(resu,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'active',int(gate.sum()),'assets',15,'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-06-10')]:
 z=x.loc[a:b]; print('regime',a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 y=R.rolling(h).sum().shift(-h); qx=[]
 for dt in R.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: qx.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(qx),np.mean(qx),np.mean(qx)/np.std(qx,ddof=1))
f.to_csv('scripts/miner_3_20330610_extreme_stress90_revalidation_signal.csv')
