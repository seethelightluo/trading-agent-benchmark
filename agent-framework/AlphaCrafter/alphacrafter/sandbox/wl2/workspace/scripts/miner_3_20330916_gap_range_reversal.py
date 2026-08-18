import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>120:return d
  except Exception: pass
xs={s:fetch(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
# One interpretable idea: lagged overnight gap reversal, scaled by recent volatility
# and activated by an unusually wide prior-day range. All inputs are lagged at signal use.
def panel(h):
 out=[]
 for s,d in xs.items():
  c=d.close.astype(float); o=d.open.astype(float); hi=d.high.astype(float); lo=d.low.astype(float)
  rv=np.log(c/c.shift(1)).rolling(20).std()
  gap=np.log(o/c.shift(1))
  rng=np.log(hi/lo)
  rz=(rng-rng.rolling(40).mean())/rng.rolling(40).std()
  # reversal: buy overnight losers / sell winners; range expansion strengthens signal
  f=(-gap/rv*(1+0.5*rz.clip(-1,2))).clip(-6,6).shift(1)
  r=c.shift(-h)/c-1
  out.append(pd.DataFrame({'date':d.date,'f':f,'r':r,'s':s}).dropna())
 x=pd.concat(out); rows=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: rows.append((dt,g.f.corr(g.r,method='spearman'),len(g)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [1,3,5,10]:
 q=panel(h); print('h=%d dates=%d avg_n=%.3f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
 q=panel(1); z=q.loc[a:b]; print('regime=%s-%s dates=%d IC=%.6f ICIR=%.6f'%(a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan))
q=panel(1); q.to_csv('scripts/miner_3_20330916_gap_range_reversal_ic.csv')
print('assets=%d total_rows=%d'%(len(xs),sum(len(d) for d in xs.values())))
