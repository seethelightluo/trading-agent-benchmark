import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   d=f(s,days=5000)
   if d is not None and len(d)>100:return d
  except:pass
D={}
for s in U:
 d=load(s)
 if d is not None:D[s]=pd.Series(d.close.values,index=pd.to_datetime(d.date))
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff(); down=lr.where(lr<0).rolling(40).std(); sig=(np.log(p/p.shift(20))/(down+1e-8)).shift(1)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna(); ns.append(len(z))
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a);print(h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('assets',len(D),'coverage',round(sig.notna().sum().sum()/(len(p)*len(D)),4),'rows',len(p),'end',p.index[-1])
