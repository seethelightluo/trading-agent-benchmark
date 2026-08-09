import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def calc(df):
 d=df.sort_values('date').reset_index(drop=True); c=d.close.astype(float).to_numpy(); r=c[1:]/c[:-1]-1
 # trend efficiency: directional displacement relative to path length, signed by direction
 out=np.full(len(c),np.nan); out[10:]=((c[10:]/c[:-10]-1)/(np.maximum(np.abs(r[-10:]).reshape(-1,1).sum(axis=1) if False else 1,1)))
 # above vector shortcut replaced below
 for i in range(10,len(c)):
  path=np.abs(r[i-10:i]).sum()
  out[i]=(c[i]/c[i-10]-1)/max(path,1e-8)
 return pd.Series(out,index=d.date)

frames={}
for s in U:
 df=get_stock_daily_data(s,days=2100)
 if df is not None and len(df)>100:
  x=calc(df); c=df.set_index('date').close.astype(float)
  fwd=c.shift(-1)/c-1
  frames[s]=pd.DataFrame({'f':x,'y':fwd})
all_dates=sorted(set().union(*[set(x.index) for x in frames.values()]))
ics=[]; counts=[]; turnovers=[]; prev=None
for dt in all_dates:
 vals=[]; ys=[]; names=[]
 for s,z in frames.items():
  if dt in z.index and np.isfinite(z.loc[dt,'f']) and np.isfinite(z.loc[dt,'y']): vals.append(z.loc[dt,'f']);ys.append(z.loc[dt,'y']);names.append(s)
 if len(vals)>=8:
  ic=pd.Series(vals).corr(pd.Series(ys),method='spearman')
  if np.isfinite(ic): ics.append(ic);counts.append(len(vals))
  ranks=dict(zip(names,pd.Series(vals).rank(pct=True)))
  if prev is not None:
   common=set(ranks)&set(prev); turnovers.append(np.mean([abs(ranks[s]-prev[s]) for s in common]))
  prev=ranks
x=np.array(ics); print('factor=10d signed trend efficiency; dates',len(x),'avg_names',np.mean(counts),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'turnover',np.mean(turnovers),'coverage',np.mean(counts)/15)
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=[v for d,v in zip([pd.Timestamp(d) for d in all_dates if any(True for _ in [0])],[])]
 sel=[v for d,v in zip([d for d in all_dates if d in all_dates],ics) if a<=str(d)[:4]<=b]
 if sel: print(a,b,len(sel),np.mean(sel),np.mean(sel)/np.std(sel,ddof=1))
for h in [3,5,10]:
 # rerun same factor against h-day forward
 zics=[]
 for dt in all_dates:
  vs=[];ys=[]
  for s,z in frames.items():
   if dt in z.index:
    # use close via y one-day unavailable; reconstruct fwd from df
    pass
 print('decay horizon',h,'not computed')
