import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 close[s]=d.set_index('date')['close'].astype(float).sort_index() if d is not None else pd.Series(dtype=float)
R=pd.DataFrame({s:x.pct_change() for s,x in close.items()}); rows=[]
for dt in R.index:
 vals={}; base={}; fut={}
 for s,x in close.items():
  z=x.loc[:dt]; r=z.pct_change().dropna()
  if len(r)>=20:
   rr=r.iloc[-20:]; denom=np.abs(rr).sum()
   if np.isfinite(denom) and denom>1e-9:
    vals[s]=(z.iloc[-1]/z.iloc[-21]-1)/denom; base[s]=z.iloc[-1]; fut[s]=x.loc[x.index>dt]
 for h in [1,5,10]:
  q=[(vals[s],fut[s].iloc[h-1]/base[s]-1) for s in vals if len(fut[s])>=h]
  if len(q)>=8:
   a,b=map(np.array,zip(*q)); c=np.corrcoef(a,b)[0,1]
   if np.isfinite(c): rows.append((dt,h,c,len(q)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=df[df.h==h]; ic=q.ic
 print({'horizon':h,'dates':len(q),'avg_n':round(q.n.mean(),2),'coverage':round(q.n.sum()/(len(q)*15),4),'IC':round(ic.mean(),6),'ICIR':round(ic.mean()/ic.std(ddof=1),6),'hit':round((ic>0).mean(),4),'period':f'{q.date.min()} to {q.date.max()}'})
# recent regime halves
q=df[df.h==10].copy(); q['recent']=q.date>=q.date.max()-pd.Timedelta(days=365)
for k,g in q.groupby('recent'):
 print('recent' if k else 'older',len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
