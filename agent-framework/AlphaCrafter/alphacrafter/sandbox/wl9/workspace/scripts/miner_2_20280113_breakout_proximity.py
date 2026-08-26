import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=3000) for s in U}
close={s:(d.set_index('date')['close'].astype(float).sort_index() if d is not None else pd.Series(dtype=float)) for s,d in px.items()}
all_dates=sorted(set().union(*[set(x.index) for x in close.values()])); rows=[]
for dt in all_dates:
 vals={}; base={}; futs={}
 for s,x in close.items():
  z=x.loc[:dt]
  if len(z)>=61:
   vals[s]=np.log(z.iloc[-1]/z.iloc[-61:-1].max()); base[s]=z.iloc[-1]
   futs[s]=x.loc[x.index>dt]
 for h in [1,5,10]:
  rr=[(f,float(fut.iloc[h-1]/base[s]-1),s) for s,f in vals.items() if len(fut:=futs[s])>=h]
  if len(rr)>=8:
   a=np.array([q[0] for q in rr]); b=np.array([q[1] for q in rr])
   if np.std(a)>0 and np.std(b)>0: rows.append((dt,h,np.corrcoef(a,b)[0,1],len(rr)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=df[df.h==h]; ic=q.ic
 print({'horizon':h,'dates':len(q),'avg_n':round(q.n.mean(),2),'coverage':round(q.n.sum()/(len(q)*15),4),'IC':round(ic.mean(),6),'ICIR':round(ic.mean()/ic.std(ddof=1),6),'hit':round((ic>0).mean(),4),'period':f"{q.date.min()} to {q.date.max()}"})
q=df[df.h==1].copy(); q['year']=pd.to_datetime(q.date).dt.year
print('yearly',q.groupby('year').ic.agg(['mean','count']).round(5).to_dict('index'))
