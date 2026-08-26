import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:x=get_index_daily_data(s,days=6000)
 except:x=None
 if x is None:
  try:x=get_stock_daily_data(s,days=6000)
  except:x=None
 if x is not None and len(x):D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff();v=r.rolling(60,min_periods=30).std()*np.sqrt(252);f=(-r.rolling(20,min_periods=15).sum()/v).shift(1).rolling(3,min_periods=2).mean();fw=np.log(p.shift(-10)/p);a=[]
for d in f.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:a.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(a,columns=['date','ic','n']).set_index('date');print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/len(D));print('ic',a.ic.mean(),'icir',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean());
for n in [365,750,1260]:q=a.tail(n);print('recent',n,q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1));print('period',a.index.min(),a.index.max());f.stack().rename('signal').to_csv('scripts/miner_1_20340302_volnorm_trend20_60vol_signal.csv');a.to_csv('scripts/miner_1_20340302_volnorm_trend20_60vol_ic.csv')
