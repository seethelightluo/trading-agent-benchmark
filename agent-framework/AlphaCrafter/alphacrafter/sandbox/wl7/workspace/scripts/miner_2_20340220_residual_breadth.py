import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
try:
 w=get_account_dict().get('watch_list',[])
 if set(U).issubset(set(w)): U=w
except Exception: pass
D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: d=fn(s,days=3000)
  except Exception: pass
  if d is not None and len(d): break
 if d is not None and len(d): D[s]=d[['date','close']].assign(date=lambda x:pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); raw=p.pct_change(20); mkt=raw.mean(axis=1); breadth=(raw>0).mean(axis=1); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
sig=((raw.sub(mkt,axis=0))/(vol+1e-8)).mul((2*breadth-1),axis=0).shift(1).rank(axis=1,pct=True)
for h in (1,5,10):
 f=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q);ds.append(dt);ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-26','2023','2026-12-31'),('2027-30','2027','2030-12-31'),('2031-34','2031','2034-12-31')]:
   q=a[[pd.Timestamp(lo)<=x<=pd.Timestamp(hi) for x in ds]]; print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
  t=[]
  for i in range(1,len(sig)):
   c=sig.iloc[i].dropna().index.intersection(sig.iloc[i-1].dropna().index)
   if len(c)>=8:t.append(np.mean(abs(sig.iloc[i][c]-sig.iloc[i-1][c])))
  print('TURN',round(np.mean(t),6),'COVERAGE',round(sig.notna().sum().sum()/(sig.shape[0]*len(U)),4),'ASSETS',len(D),'RANGE',p.index.min(),p.index.max())
sig.to_csv('scripts/miner_2_20340220_residual_breadth_signal.csv',index_label='date')
