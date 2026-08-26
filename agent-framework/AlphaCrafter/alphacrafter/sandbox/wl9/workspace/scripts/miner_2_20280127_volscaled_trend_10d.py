import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=3000) for s in U}
close={s:(d.set_index('date')['close'].astype(float).sort_index() if d is not None else pd.Series(dtype=float)) for s,d in px.items()}
R=pd.DataFrame({s:x.pct_change() for s,x in close.items()}); rows=[]
for dt in sorted(R.index):
 vals={}; base={}; future={}
 for s,x in close.items():
  z=x.loc[:dt]; ret=z.pct_change()
  if len(z)>=22:
   vol=ret.iloc[-20:].std(ddof=1)
   if np.isfinite(vol) and vol>1e-8: vals[s]=(z.iloc[-1]/z.iloc[-21]-1)/vol; base[s]=z.iloc[-1]; future[s]=x.loc[x.index>dt]
 for h in [1,5,10]:
  q=[(v,(future[s].iloc[h-1]/base[s]-1)) for s,v in vals.items() if len(future[s])>=h]
  if len(q)>=8:
   a=np.array([z[0] for z in q]); b=np.array([z[1] for z in q]); c=np.corrcoef(a,b)[0,1]
   if np.isfinite(c): rows.append((dt,h,c,len(q)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=df[df.h==h]; ic=q.ic
 print({'horizon':h,'dates':len(q),'avg_n':round(q.n.mean(),2),'coverage':round(q.n.sum()/(len(q)*15),4),'IC':round(ic.mean(),6),'ICIR':round(ic.mean()/ic.std(ddof=1),6),'hit':round((ic>0).mean(),4),'period':f'{q.date.min()} to {q.date.max()}'})
