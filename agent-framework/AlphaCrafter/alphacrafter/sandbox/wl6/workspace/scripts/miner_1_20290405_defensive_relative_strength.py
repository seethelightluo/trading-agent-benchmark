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
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
  c=pd.to_numeric(d.close,errors='coerce'); S[s]=c.pct_change()
R=pd.concat(S,axis=1).sort_index(); eq=R[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
rows=[]
for s in U:
 r=R[s]; down=eq<0
 # favor positive medium momentum, low downside beta and low idio risk
 cov=r.rolling(60,min_periods=40).cov(eq); var=eq.rolling(60,min_periods=40).var()
 db=(cov/var).where(down.rolling(60,min_periods=40).sum()>=15)
 resid=r.rolling(20,min_periods=15).sum()-eq.rolling(20,min_periods=15).sum()*db
 sig=(r.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+.01)) - .35*db - .25*resid.abs()
 for dt in R.index:
  if dt not in r.index: continue
  vals=[sig.get(dt), (r.shift(-1).rolling(1).sum().get(dt)), r.shift(-5).rolling(5).sum().get(dt),r.shift(-10).rolling(10).sum().get(dt)]
  if np.isfinite(vals).all(): rows.append((dt,s,*vals))
d=pd.DataFrame(rows,columns=['date','s','sig','f1','f5','f10'])
for h in [1,5,10]:
 out=[]
 for dt,g in d.groupby('date'):
  if len(g)>=8: out.append(g.sig.rank().corr(g[f'f{h}'].rank()))
 x=pd.Series(out).dropna(); print(f'{h}d dates={len(x)} avg_n={d.groupby("date").size().mean():.2f} coverage={d.groupby("date").size().mean()/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.4f} hit={(x>0).mean():.4f}')
