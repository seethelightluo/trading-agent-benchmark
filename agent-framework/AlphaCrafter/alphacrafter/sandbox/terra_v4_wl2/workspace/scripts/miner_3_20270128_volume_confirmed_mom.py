import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 D[s]=x
# volume-confirmed medium momentum: 20d return multiplied by log volume surprise, clipped
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); v=x.volume.replace(0,np.nan)
 mom=x.close.pct_change(20)
 vs=np.log(v/v.rolling(60,min_periods=30).median()).clip(-2,2)
 fac=mom*(1+0.5*vs)
 fr=x.close.shift(-1)/x.close-1
 z=pd.DataFrame({'f':fac,'fr':fr}).dropna(); z['sym']=s; rows.append(z.reset_index())
a=pd.concat(rows)
piv=a.pivot(index='date',columns='sym',values='f'); ret=a.pivot(index='date',columns='sym',values='fr')
ics=[]; counts=[]
for dt in piv.index:
 z=pd.concat([piv.loc[dt],ret.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); counts.append(len(z))
ics=np.array(ics)
print('dates',len(ics),'avg_names',np.mean(counts),'IC',np.mean(ics),'ICIR',np.mean(ics)/np.std(ics,ddof=1),'hit',np.mean(ics>0),'coverage',sum(counts)/(len(ics)*15))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2026-07-16','2027-01-27')]:
 q=[]
 for dt in piv.loc[lo:hi].index:
  z=pd.concat([piv.loc[dt],ret.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print(lo,hi,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
# artifact
out=piv.stack().rename('signal').reset_index(); out.to_csv('../persistent/factor_signals_miner_3_20270128_volume_confirmed_mom.csv',index=False)
