import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None and len(d): return d
        except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); c=d.close
 r=c.pct_change(); down=r.clip(upper=0)
 # downside-risk adjusted medium momentum, with a mild consistency multiplier
 dd=np.sqrt((down.pow(2)).rolling(40).mean())
 mom=c.pct_change(20)
 consistency=(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True)-.5)*2
 S[s]=pd.DataFrame({'sig':mom/(dd*np.sqrt(20))* (0.5+0.5*consistency),
                    'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
def run(col):
 rows=[]
 for dt in sorted(set().union(*[x.index for x in S.values()])):
  vals=[x.loc[dt] for x in S.values() if dt in x.index and np.isfinite(x.loc[dt,[col,'f10']]).all()]
  if len(vals)>=8:
   z=pd.DataFrame(vals); rows.append((dt,z[col].rank().corr(z.f10.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('10d dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4),'coverage',round(q.n.mean()/15,4),'assets',len(S))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,len(v),round(v.mean(),6) if len(v) else None)
 for h in [1,5]:
  rr=[]
  for dt in sorted(set().union(*[x.index for x in S.values()])):
   vals=[x.loc[dt] for x in S.values() if dt in x.index and np.isfinite(x.loc[dt,[col,f'f{h}']]).all()]
   if len(vals)>=8:
    z=pd.DataFrame(vals);rr.append(z[col].rank().corr(z[f'f{h}'].rank()))
  a=pd.Series(rr);print(h,'d IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),4),'dates',len(a))
run('sig')
