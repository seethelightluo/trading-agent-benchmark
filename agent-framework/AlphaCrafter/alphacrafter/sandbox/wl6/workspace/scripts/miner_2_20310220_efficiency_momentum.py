import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
S={}
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index(); c=d.close; r=c.pct_change()
 # Interpretable trend efficiency: net 60-session return divided by path length, lagged.
 eff=c.pct_change(60)/(r.abs().rolling(60).sum())
 S[s]=pd.DataFrame({'sig':eff.shift(1),**{f'f{h}':c.shift(-h)/c-1 for h in [5,10,20]}})
D=sorted(set().union(*[x.index for x in S.values()]))
print('assets',len(S),'dates',len(D),'span',D[0].date(),D[-1].date())
for h in [5,10,20]:
 rows=[]
 for dt in D:
  a=[(x.loc[dt].sig,x.loc[dt][f'f{h}']) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt][f'f{h}'])]
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['s','r']); rows.append((dt,z.s.corr(z.r,method='spearman'),len(a)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,5),'IC',round(m,8),'daily_ICIR',round(m/sd,8),'hit',round((q.ic>0).mean(),5))
 print('years',q.assign(y=q.date.dt.year).groupby('y').ic.mean().round(4).to_dict())
r=pd.DataFrame({k:v.sig.rank(pct=True) for k,v in S.items()}); print('turnover_proxy',round(r.diff().abs().mean(axis=1).mean(),6))
