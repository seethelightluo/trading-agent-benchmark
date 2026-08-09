import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(P).sort_index(); r=np.log(p).diff(); r5=r.rolling(5,min_periods=4).sum(); r20=r.rolling(20,min_periods=15).sum(); base=(r5-r20/4).shift(1)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill(); vr=np.log(v).diff();
for name,mask in [('all',pd.Series(True,index=p.index)),('vix_down',vr.shift(1)<0),('vix_up',vr.shift(1)>0),('vix_calm',v.shift(1)<v.shift(1).rolling(60,min_periods=30).median()),('vix_stress',v.shift(1)>v.shift(1).rolling(60,min_periods=30).median())]:
 f=base.where(mask, np.nan); y=p.shift(-1)/p-1; out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(out);print(name,'dates',len(s),'meanN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean(),'coverage',f.notna().stack().mean())
# sign-flipped regime interaction, always populated where base exists
for name,mult in [('vix_sign',np.where(vr.shift(1)<0,1,-1)),('stress_sign',np.where(v.shift(1)>v.shift(1).rolling(60,min_periods=30).median(),-1,1))]:
 f=base.mul(mult,axis=0);y=p.shift(-1)/p-1;out=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(out);print(name,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
