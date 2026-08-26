import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=3000) for s in U}; close={s:(d.set_index('date')['close'].astype(float).sort_index() if d is not None else pd.Series(dtype=float)) for s,d in px.items()}; R=pd.DataFrame({s:x.pct_change() for s,x in close.items()}); disp=R.std(axis=1); med=disp.rolling(60,min_periods=30).median(); rows=[]
for dt in sorted(set(R.index)):
 if pd.isna(disp.get(dt)) or pd.isna(med.get(dt)) or disp[dt]<=med[dt]: continue
 vals={}; bases={}; futs={}
 for s,x in close.items():
  z=x.loc[:dt]
  if len(z)>=2: vals[s]=-(z.iloc[-1]/z.iloc[-2]-1);bases[s]=z.iloc[-1];futs[s]=x.loc[x.index>dt]
 for h in [1,5]:
  rr=[(f,float(fut.iloc[h-1]/bases[s]-1),s) for s,f in vals.items() if len(fut:=futs[s])>=h]
  if len(rr)>=8:
   a=np.array([q[0] for q in rr]);b=np.array([q[1] for q in rr]);rows.append((dt,h,np.corrcoef(a,b)[0,1],len(rr)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5]:
 q=df[df.h==h];ic=q.ic;print({'horizon':h,'dates':len(q),'avg_n':round(q.n.mean(),2),'coverage':round(q.n.sum()/(len(q)*15),4),'IC':round(ic.mean(),6),'ICIR':round(ic.mean()/ic.std(ddof=1),6),'hit':round((ic>0).mean(),4),'period':f'{q.date.min()} to {q.date.max()}'})
