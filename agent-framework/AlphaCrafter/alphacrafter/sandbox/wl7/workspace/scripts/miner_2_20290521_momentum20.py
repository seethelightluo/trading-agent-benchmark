import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 try:d=get_index_daily_data(s,2600)
 except:d=None
 if d is None:
  try:d=get_stock_daily_data(s,2600)
  except:d=None
 if d is not None:fs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(fs).sort_index().ffill(); f=p.shift(1)/p.shift(21)-1; ics=[]; rows=[]
for i in range(21,len(p)-10):
 z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));rows.append([p.index[i],len(z),ics[-1]])
a=np.array(ics); print('raw 20d momentum','assets',len(fs),'dates',len(a),'avg_n',np.mean([x[1] for x in rows]),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0));
for q in [a[pd.to_datetime([x[0] for x in rows])>='2028-09-01']]:print('recent',len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
pd.DataFrame(rows,columns=['date','n','ic']).to_csv('scripts/miner_2_20290521_momentum20_signal.csv',index=False)
