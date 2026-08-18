import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in [get_index_daily_data,get_stock_daily_data]:
  try:
   d=f(s,days=5000)
   if d is not None and len(d)>100:return d
  except Exception: pass
xs={s:get(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
# Intraday capitulation: weak close-to-open return amplified by abnormal volume,
# lagged one day to ensure observable information at decision time.
def calc(h):
 rows=[]
 for s,d in xs.items():
  c=d.close.astype(float); o=d.open.astype(float); vol=d.volume.astype(float)
  rv=np.log(c/c.shift(1)).rolling(20).std()
  intr=np.log(c/o)
  vz=np.log1p(vol).sub(np.log1p(vol).rolling(40).mean()).div(np.log1p(vol).rolling(40).std())
  # negative intraday selloff, with volume confirmation, bounded for robustness
  f=(-intr/rv*(1+0.35*vz.clip(-1,3))).clip(-6,6).shift(1)
  r=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'date':d.date,'f':f,'r':r,'s':s}).dropna())
 x=pd.concat(rows); out=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   out.append((dt,g.f.corr(g.r,method='spearman'),len(g)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
 return q
for h in [1,3,5,10]:
 q=calc(h); print(f'h={h} dates={len(q)} avg_n={q.n.mean():.3f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
 q=calc(1); z=q.loc[a:b]; print(f'regime={a}-{b} dates={len(z)} IC={z.ic.mean():.6f} ICIR={z.ic.mean()/z.ic.std(ddof=1):.6f}')
# signal artifact for provenance
q=calc(1); q.to_csv('scripts/miner_3_20330902_intraday_capitulation_ic.csv')
print('assets',len(xs), 'coverage observations', sum(len(d) for d in xs.values()))
