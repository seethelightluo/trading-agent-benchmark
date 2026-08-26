import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try: x=get_index_daily_data(s,days=6000)
 except Exception: x=None
 if x is None:
  try:x=get_stock_daily_data(s,days=6000)
  except Exception:x=None
 if x is not None and len(x):
  x=x.copy();x['date']=pd.to_datetime(x['date']);D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
f=(-(r.rolling(20,min_periods=15).sum()/vol)).shift(1).rolling(3,min_periods=2).mean(); fwd=np.log(p.shift(-10)/p); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); out={}
for h in [1,5,10,20]:
 fw=np.log(p.shift(-h)/p); q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 out[h]=float(np.nanmean(q))
print('instruments',len(D),list(D));print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*len(D)));print('ic',a.ic.mean(),'icir',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean());print('decay',out)
for n in [365,750,1260]:
 q=a.tail(n);print('recent',n,q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),len(q))
print('period',a.index.min(),a.index.max());f.stack().rename('signal').to_csv('scripts/miner_1_20340302_volnorm_trend20_signal.csv');a.to_csv('scripts/miner_1_20340302_volnorm_trend20_ic.csv')
