import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>150:return d
  except: pass
xs={s:fetch(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float)
vg=(v>v.rolling(20).mean()).shift(1)
def panel(h):
 rows=[]
 for s,d in xs.items():
  d=d.copy(); d.date=pd.to_datetime(d.date); c=d.close.astype(float); lr=np.log(c/c.shift(1)); vol=lr.rolling(20).std()
  f=(np.log(c/c.shift(10))/vol).shift(1); r=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'date':d.date,'raw':f,'r':r,'s':s}).dropna())
 x=pd.concat(rows); out=[]
 for dt,g in x.groupby('date'):
  z=g.raw-g.raw.median(); gate=vg.get(dt,np.nan)
  if len(g)>=8 and z.nunique()>1 and g.r.nunique()>1 and pd.notna(gate):
   sig=-z if gate else z
   out.append((dt,sig.corr(g.r,method='spearman'),len(g),int(gate)))
 return pd.DataFrame(out,columns=['date','ic','n','gate']).set_index('date')
for h in [1,3,5,10]:
 q=panel(h); print('h=%d dates=%d avg_n=%.3f IC=%.6f ICIR=%.6f hit=%.4f gate=%.3f'%(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean(),q.gate.mean()))
for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
 q=panel(10); z=q.loc[a:b]; print('regime=%s-%s dates=%d IC=%.6f ICIR=%.6f'%(a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
