import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
for w in [30,40,50,60,80]:
 rows=[]
 for i,t in enumerate(p.index):
  if i<w+2 or i+20>=len(p): continue
  sig=-(p.iloc[i]/p.iloc[i-w]-1)*(r.iloc[i-w+1:i+1]>0).mean()
  f=p.shift(-20).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=np.array(rows); print('window',w,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
