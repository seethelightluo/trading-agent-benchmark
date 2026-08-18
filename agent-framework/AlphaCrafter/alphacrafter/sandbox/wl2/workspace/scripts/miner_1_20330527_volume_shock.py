import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-05-27')
P={};V={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=cut]; P[s]=d.close; V[s]=d.volume
p=pd.DataFrame(P); v=pd.DataFrame(V); r=np.log(p).diff()
# lagged signed volume shock: reversal after unusually large volume and return, normalized by vol
z=(v/v.rolling(20).mean()-1).shift(1); fac=(-r.shift(1)*z)/(r.rolling(20).std().shift(1))
for h in [1,3,5,10]:
 vals=[];ns=[]; fr=np.log(p).shift(-h)-np.log(p)
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic);ns.append(len(a))
 x=np.array(vals);print(h,len(x),round(np.mean(ns),2),round(np.mean(ns)/15,3),round(np.mean(x),6),round(np.mean(x)/np.std(x,ddof=1),6),round(np.mean(x>0),4))
fac.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330527_volume_shock_signal.csv',index=False)
