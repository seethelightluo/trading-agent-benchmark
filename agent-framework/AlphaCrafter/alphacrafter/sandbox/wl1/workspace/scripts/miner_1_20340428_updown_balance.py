import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date'); frames[s]=d['close'].astype(float).rename(s)
px=pd.concat(frames.values(),axis=1).sort_index(); r=np.log(px).diff()
# upside/downside RMS balance, lagged, cross-sectional factor
up=r.clip(lower=0).rolling(40).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
dn=(-r.clip(upper=0)).rolling(40).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
f=(up/(dn+1e-10)).shift(1)
rows=[]
for i in range(len(px)-10):
 dt=px.index[i]; vals=f.iloc[i]; fr=np.log(px.iloc[i+10]/px.iloc[i]); z=pd.concat([vals,fr],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15); print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=x.loc[a:b]; print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan)
for h in [5,10,20]:
 rr=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.mean(rr),len(rr))
f.to_csv('scripts/miner_1_20340428_updown_balance_signal.csv')
