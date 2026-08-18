import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-11-14'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d[d.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v[v.date<=cut].set_index('date').close.reindex(P.index).ffill(); vz=((v-v.rolling(60,min_periods=30).mean())/(v.rolling(60,min_periods=30).std()+1e-8)).clip(-3,3)
sig=(-P.pct_change(5)/(r.clip(upper=0).abs().rolling(10,min_periods=8).mean()+1e-8)).mul((1+.5*np.maximum(vz,0)),axis=0).shift(1)
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; a=[];ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);print(h,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),6),round(np.mean(a>0),4))
print('turn',round(sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
