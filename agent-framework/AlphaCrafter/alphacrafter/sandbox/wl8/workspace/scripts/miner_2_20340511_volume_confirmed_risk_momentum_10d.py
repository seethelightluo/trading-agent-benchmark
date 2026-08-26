import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-05-11'); base='../persistent/stock_data'; px={}; vol={}
for s in SYMS:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); d=d.loc[d.index<=cut]
 px[s]=d.close; vol[s]=d.volume.replace(0,np.nan)
prices=pd.DataFrame(px); volumes=pd.DataFrame(vol)
ret20=prices/prices.shift(20)-1; rv=prices.pct_change().rolling(20,min_periods=15).std()
vr=(volumes.rolling(20,min_periods=15).mean()/volumes.rolling(60,min_periods=40).mean()).clip(.25,4)
f=((ret20/rv)*np.log(vr)).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(prices)-10):
 x=f.iloc[i]; y=prices.iloc[i+10]/prices.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((prices.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); rank=f.rank(axis=1,pct=True); to=rank.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(r),'period',r.index.min().date(),r.index.max().date(),'avgN',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'turnover',to)
for n in [365,750,1260]:
 q=r.tail(n); print('recent',n,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
for h in [1,5,20]:
 rr=[]
 for i in range(len(prices)-h):
  z=pd.concat([f.iloc[i],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),'n',len(rr))
out='scripts/miner_2_20340511_volume_confirmed_risk_momentum_10d'; r.to_csv(out+'_ic.csv'); f.to_csv(out+'_signal.csv'); print('artifacts',out+'_ic.csv',out+'_signal.csv')
