import pandas as pd,numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-04-14');A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index();P=P[P.index<=CUT]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill();stress=(v/v.rolling(60,min_periods=40).median()).clip(.5,2)
f=-(P/P.shift(20)-1)*stress.values[:,None]
for h in [5,10,20]:
 z=[];cov=[]
 for i in range(len(P)-h):
  x=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.x,x.y).statistic);cov.append(len(x)/15)
 z=np.array(z);print(h,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4),round(np.mean(cov),4))
