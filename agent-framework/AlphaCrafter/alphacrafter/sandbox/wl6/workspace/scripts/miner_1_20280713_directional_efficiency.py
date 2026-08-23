import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None:return d
        except Exception: pass
    return None
ss={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index(); r=np.log(d.close).diff()
 # lagged directional efficiency: net movement relative to path length, robust to asset volatility
 eff=r.rolling(20).sum()/r.abs().rolling(20).sum()
 # require completed day: all features are at date, forward starts next day
 ss[s]=pd.DataFrame({'sig':eff,'f1':d.close.pct_change().shift(-1),'f5':d.close.shift(-5)/d.close-1,'f10':d.close.shift(-10)/d.close-1})
print('assets',len(ss))
def run(h):
 rows=[]
 for dt in sorted(set().union(*[x.index for x in ss.values()])):
  z=[]
  for s,x in ss.items():
   if dt in x.index and np.isfinite(x.at[dt,'sig']) and np.isfinite(x.at[dt,h]):z.append((x.at[dt,'sig'],x.at[dt,h]))
  if len(z)>=8:
   q=pd.DataFrame(z,columns=['a','b']); rows.append((dt,q.a.rank().corr(q.b.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1); ir=m/sd*np.sqrt(252)
 print(h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(m,7),'ICIR',round(ir,5),'hit',round((q.ic>0).mean(),4))
 for lab,a,b in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026'),('2027-28','2027','2028')]:
  v=q[(q.date.dt.year>=int(a))&(q.date.dt.year<=int(b))].ic
  print(lab,len(v),round(v.mean(),7) if len(v) else None)
run('f1');run('f5');run('f10')
# rank turnover
prev=None;ts=[]
for dt in sorted(set().union(*[x.index for x in ss.values()])):
 v={s:x.at[dt,'sig'] for s,x in ss.items() if dt in x.index and np.isfinite(x.at[dt,'sig'])}
 if len(v)>=8:
  rk=pd.Series(v).rank();
  if prev is not None:
   c=list(set(rk.index)&set(prev.index));ts.append(np.mean(abs(rk[c]-prev[c])/(len(c)-1 or 1)))
  prev=rk
print('coverage',sum(len(x.dropna()) for x in ss.values())/(len(ss)*max(len(x) for x in ss.values())),'turnover',np.mean(ts),'dates',len(set().union(*[x.index for x in ss.values()])))
