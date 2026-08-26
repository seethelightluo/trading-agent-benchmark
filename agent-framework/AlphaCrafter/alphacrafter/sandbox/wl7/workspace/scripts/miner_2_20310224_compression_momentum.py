import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            d=fn(s, days=5000)
            if d is not None and len(d)>100: return d
        except Exception: pass
    return None

def rankic(a,b):
    z=pd.DataFrame({'a':a,'b':b}).dropna()
    return z.a.rank().corr(z.b.rank()) if len(z)>=8 else np.nan
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# align dates and compute factor using completed t; evaluate return t+1 etc
rows=[]
for s,d in D.items():
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date')
 c=d.close.astype(float); r=c.pct_change(); rng=(d.high-d.low)/c
 # compression relative to trailing range, breakout continuation but standardized
 mom=r.rolling(5).sum()/r.rolling(20).std()
 compression=(rng.rolling(5).mean()/rng.rolling(20).mean()).clip(0.25,3)
 f=mom*(1.5-compression) # favor recent momentum following compressed ranges
 # lag implicitly factor at t predicts t+1
 for i in range(len(d)-1): rows.append((d.date.iloc[i],s,f.iloc[i],r.iloc[i+1]))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
out=[]
for dt,g in x.groupby('date'):
 if g.factor.notna().sum()>=8: out.append((dt,rankic(g.factor,g.fwd),g.factor.notna().mean(),g.factor.rank().corr(g.factor.shift(1).rank()) if False else np.nan))
o=pd.DataFrame(out,columns=['date','ic','coverage','dummy']).dropna(subset=['ic'])
ics=o.ic
print('dates',len(o),'avg_n',x.groupby('date').factor.apply(lambda z:z.notna().sum()).mean(),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'coverage',o.coverage.mean())
for n in [252,756,1500]:
 q=ics.tail(n); print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for h in [3,5,10,20]:
 rr=[]
 for s,d in D.items():
  d=d.sort_values('date'); r=d.close.pct_change(h).shift(-h)
  c=d.close.astype(float); ret=c.pct_change(); rng=(d.high-d.low)/c
  f=(ret.rolling(5).sum()/ret.rolling(20).std())*(1.5-(rng.rolling(5).mean()/rng.rolling(20).mean()).clip(.25,3))
  z=pd.DataFrame({'date':d.date,'f':f,'r':r}).dropna()
  rr.extend((dt,rankic(g.f,g.r)) for dt,g in z.groupby('date') if g.f.notna().sum()>=8)
 q=pd.DataFrame(rr,columns=['d','ic']).dropna(); print('horizon',h,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
# signal artifact for provenance
sig=[]
for dt,g in x.groupby('date'):
 for _,z in g.iterrows(): sig.append((dt,z.symbol,z.factor))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310224_compression_momentum_signal.csv',index=False)
ics.to_csv('scripts/miner_2_20310224_compression_momentum_ic.csv',index=False)
