import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();m=r.mean(axis=1)
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date')['close'].reindex(p.index).ffill()
for h in [5,20]:
 rows=[]
 for i,t in enumerate(p.index):
  if i<150 or i+h>=len(p) or v.iloc[i]<=v.iloc[i-120:i].quantile(.75):continue
  rr=r.iloc[i-59:i+1];mm=m.iloc[i-59:i+1];b=rr.apply(lambda x:x.cov(mm)/(mm.var()+1e-12));ret=p.iloc[i]/p.iloc[i-5]-1;sig=-(ret-b*ret.mean())/(rr.iloc[-10:].std()+1e-8);f=p.iloc[i+h]/p.iloc[i]-1;q=pd.concat([sig,f],axis=1).dropna();
  if len(q)>=8:rows.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=pd.Series(rows);print('h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
