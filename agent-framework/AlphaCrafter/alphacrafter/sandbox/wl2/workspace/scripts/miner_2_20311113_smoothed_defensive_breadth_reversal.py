import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
pd_=pd.DataFrame(D).sort_index().ffill(); r=pd_.pct_change(); vol=r.rolling(20).std()
defs=['XAU','US10Y','CN10Y']; risk=['SPX','NDX','SOX','000300.SH','SX5E','HSI','N225','BTC','ETH','COPPER','WTI']
# Continuous defensive leadership, with a slower 20-day smoothed risk breadth weakness transform.
deflead=r[defs].rolling(10).mean().mean(axis=1)-r[risk].rolling(10).mean().mean(axis=1)
risk_breadth=(r[risk].rolling(10).mean()>0).mean(axis=1).rolling(20).mean()
intensity=(deflead.clip(lower=0)/0.01).clip(upper=1)*np.sqrt((1-risk_breadth).clip(lower=0))
f=-(r.rolling(3).sum()/vol).mul(intensity,axis=0)
rows=[]
for i in range(len(pd_)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c): rows.append((pd_.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def ir(x): return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
print('dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'nonzero',round((intensity>0).mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic; print(a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z) else None)
for h in [1,3,5,10]:
 y=pd_.pct_change(h).shift(-h); rr=[]
 for i in range(len(pd_)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6) if rr else None)
f.to_csv('scripts/miner_2_20311113_smoothed_defensive_breadth_reversal_signal.csv')
