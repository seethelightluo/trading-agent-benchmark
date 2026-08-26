import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>100:return d
  except:pass
D={s:fetch(s) for s in U};D={s:d.sort_values('date').drop_duplicates('date') for s,d in D.items() if d is not None}
def ic(a,b):
 z=pd.DataFrame({'a':a,'b':b}).dropna();return z.a.rank().corr(z.b.rank()) if len(z)>=8 else np.nan
rows=[]
for s,d in D.items():
 c=d.close.astype(float);r=c.pct_change();m=r.rolling(20).sum()/r.rolling(40).std();cons=(r.gt(0).rolling(20).mean()-.5)*2;f=m*cons
 for i in range(len(d)-1):rows.append((pd.to_datetime(d.date.iloc[i]),s,f.iloc[i],r.iloc[i+1]))
x=pd.DataFrame(rows,columns=['date','symbol','f','fr']);res=[]
for dt,g in x.groupby('date'):
 if g.f.notna().sum()>=8:res.append((dt,ic(g.f,g.fr),g.f.notna().sum()))
o=pd.DataFrame(res,columns=['date','ic','n']).dropna();q=o.ic
print('dates',len(o),'avg_n',o.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',x.f.notna().mean())
for n in [252,756,1500]:
 z=q.tail(n);print('recent',n,z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
for h in [3,5,10,20]:
 rr=[]
 for s,d in D.items():
  c=d.close.astype(float);r=c.pct_change();m=r.rolling(20).sum()/r.rolling(40).std();f=m*(r.gt(0).rolling(20).mean()-.5)*2;fr=c.pct_change(h).shift(-h)
  for i in range(len(d)-h):rr.append((pd.to_datetime(d.date.iloc[i]),f.iloc[i],fr.iloc[i]))
 z=pd.DataFrame(rr,columns=['date','f','r']);vals=[ic(g.f,g.r) for _,g in z.groupby('date') if g.f.notna().sum()>=8];vals=pd.Series(vals).dropna();print('h',h,'dates',len(vals),'IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1))
pd.DataFrame([(a,b,c) for a,b,c in zip(x.date,x.symbol,x.f)],columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310224_consistency_signal.csv',index=False);o[['date','ic']].to_csv('scripts/miner_2_20310224_consistency_ic.csv',index=False)
