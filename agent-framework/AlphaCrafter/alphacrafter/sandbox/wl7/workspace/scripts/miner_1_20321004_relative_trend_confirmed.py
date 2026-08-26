import numpy as np,pandas as pd,os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-10-03'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x['date']=pd.to_datetime(x['date']);x=x[x.date<=cut].sort_values('date').drop_duplicates('date');D[s]=x.set_index('date')
P=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}); R=P.pct_change(20); V=P.pct_change().rolling(60,min_periods=40).std(); short=P.pct_change(5)
rel=R.sub(R.mean(axis=1),axis=0); f=rel/(V+1e-8)*(1+0.25*np.tanh(short/(V*5+1e-8)))
for h in [1,5,10,20]:
 rows=[]
 for s in D:
  z=pd.DataFrame({'date':f.index,'f':f[s].values,'y':(P[s].shift(-h)/P[s]-1).values}).dropna();rows.append(z)
 a=pd.concat(rows,ignore_index=True); out=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>=3 and g.y.nunique()>=3: out.append((d,g.f.corr(g.y),len(g)))
 q=pd.DataFrame(out,columns=['date','ic','n']); print('H',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 if h==10:
  sig=f.stack().rename('factor').reset_index();sig.columns=['date','symbol','factor'];sig.to_csv('scripts/miner_1_20321004_relative_trend_confirmed_signal.csv',index=False)
  print('thirds',*[q.ic.iloc[i:i+len(q)//3].mean() for i in [0,len(q)//3,2*len(q)//3]])
