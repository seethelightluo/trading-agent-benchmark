import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None and len(x): D[s]=x.set_index('date')
# Close-location pressure: persistent closes near the day's extreme after a 3d move
# reversal signal; uses only completed bar and is volatility normalized.
frames=[]
for s,x in D.items():
 rng=(x.high-x.low).replace(0,np.nan)
 clv=((x.close-x.low)/rng-.5)*2
 frames.append(pd.DataFrame({'close':x.close,'clv':clv,'range':rng,'sym':s}))
a=pd.concat(frames).reset_index().pivot(index='date',columns='sym',values='close').sort_index().ffill()
clv=pd.concat(frames).reset_index().pivot(index='date',columns='sym',values='clv').reindex(a.index)
rng=pd.concat(frames).reset_index().pivot(index='date',columns='sym',values='range').reindex(a.index)
r=a.pct_change(); vol=r.rolling(20).std()
# fade 3-day directional pressure, with intraday close-location confirming exhaustion
f=-(r.rolling(3).sum()/vol)*(-clv.rolling(3).mean())
f=f.replace([np.inf,-np.inf],np.nan)
def ir(x): return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
rows=[]
for i in range(len(a)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c): rows.append((a.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('universe',len(D),'dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4))
for a0,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 z=q.loc[a0:b].ic; print('regime',a0,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z)>1 else None)
for h in [1,3,5,10]:
 y=a.pct_change(h).shift(-h); rr=[]
 for i in range(len(a)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6) if rr else None)
f.to_csv('scripts/miner_2_20320513_clv_pressure_reversal_signal.csv')
