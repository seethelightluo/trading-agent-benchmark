import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,days=4000)
 return d[['date','close']].drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.concat({s:g(s) for s in U},axis=1).sort_index().ffill();r=np.log(p).diff()
ret=r.rolling(5).sum();vol=r.rolling(20).std();disp=r.std(axis=1).rolling(20).mean()
f=(-ret.div(vol+1e-8).div(1+disp,axis=0)).shift(1)
for h in [1,3,5,10]:
 fr=np.log(p.shift(-h)/p);a=[];ds=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 a=np.array(a); dates=np.array(ds,dtype='datetime64[ns]')
 print('H',h,'obs',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-15')]:
  q=a[(dates>=np.datetime64(lo))&(dates<=np.datetime64(hi))]
  if len(q):print(' ',lo,'n',len(q),'IC',round(q.mean(),6),'IR',round(q.mean()/(q.std(ddof=1)+1e-12),6))
print('coverage',round(f.notna().mean().mean(),4),'dates',len(f))
out=f.copy();out.index.name='date';out.reset_index().to_csv('scripts/miner_1_20300516_short_reversal_signal.csv',index=False)
