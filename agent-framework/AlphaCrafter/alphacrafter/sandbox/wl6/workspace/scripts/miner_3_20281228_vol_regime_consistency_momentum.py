import numpy as np,pandas as pd
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
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();c=d.close;r=c.pct_change()
 vol=r.rolling(20).std(); mom=c.pct_change(20)
 # Trend quality: momentum per realized risk, rewarded only when recent volatility is below its 60d median
 vr=vol/vol.rolling(60).median()
 quality=mom/(vol*np.sqrt(20))
 consistency=(r.rolling(20).apply(lambda x:np.mean(x>0),raw=True)-.5)*2
 sig=quality*(.5+.5*consistency)*(1/(1+np.maximum(vr-1,0)))
 S[s]=pd.DataFrame({'sig':sig,'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
D=sorted(set().union(*[x.index for x in S.values()]))
def calc(h):
 out=[]
 for dt in D:
  vals=[x.loc[dt] for x in S.values() if dt in x.index and np.isfinite(x.loc[dt,['sig',f'f{h}']]).all()]
  if len(vals)>=8:
   z=pd.DataFrame(vals);out.append((dt,z.sig.rank().corr(z[f'f{h}'].rank()),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']);m=q.ic.mean();sd=q.ic.std(ddof=1)
 print(f'{h}d dates={len(q)} avg_n={q.n.mean():.2f} coverage={q.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(q.ic>0).mean():.4f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic;print('regime',a,len(v),f'{v.mean():.6f}')
for h in [1,5,10]:calc(h)
