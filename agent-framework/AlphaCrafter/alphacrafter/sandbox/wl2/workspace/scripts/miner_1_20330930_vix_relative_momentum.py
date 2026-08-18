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
# Candidate: cross-asset relative momentum, conditioned on lagged VIX trend.
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vix=vix.set_index('date')['close'].astype(float)
def panel(h):
 rows=[]
 for s,d in xs.items():
  d=d.copy(); d.date=pd.to_datetime(d.date); c=d.close.astype(float); lr=np.log(c/c.shift(1))
  mom=np.log(c/c.shift(10)); med=pd.Series(mom.values,index=d.date).rolling(1).mean() # placeholder cross section later
  vol=lr.rolling(20).std(); f=(mom/vol).shift(1)
  rows.append(pd.DataFrame({'date':d.date,'raw':f,'r':c.shift(-h)/c-1,'s':s}))
 x=pd.concat(rows).dropna(); x['vix']=x.date.map(vix).ffill(); x['vixma']=x.vix.rolling(20).mean() # global mapped series is imperfect
 # rank relative to date median, with VIX rising gate multiplier
 x['f']=x['raw']
 out=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.raw.nunique()>1 and g.r.nunique()>1:
   # relative cross-sectional signal; high VIX trend reverses momentum orientation
   z=g.raw-g.raw.median(); gate=g.vix.iloc[0]>g.vixma.iloc[0]
   sig=-z if gate else z
   out.append((dt,sig.corr(g.r,method='spearman'),len(g),int(gate)))
 return pd.DataFrame(out,columns=['date','ic','n','gate']).set_index('date')
for h in [1,3,5,10]:
 q=panel(h); print('h=%d dates=%d avg_n=%.3f IC=%.6f ICIR=%.6f hit=%.4f gate=%.3f'%(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean(),q.gate.mean()))
q=panel(1)
for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
 z=q.loc[a:b]; print('regime=%s-%s dates=%d IC=%.6f ICIR=%.6f'%(a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
