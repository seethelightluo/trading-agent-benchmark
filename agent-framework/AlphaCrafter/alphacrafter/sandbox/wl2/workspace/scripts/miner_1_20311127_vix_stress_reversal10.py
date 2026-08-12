import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,3000)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
pd_=pd.DataFrame(D).sort_index().ffill();r=pd_.pct_change();v=r.rolling(20).std();r10=pd_.pct_change(10)
vx=pd.read_csv('../persistent/index_data/VIX.csv');vx.date=pd.to_datetime(vx.date);vc=vx.set_index('date').close.astype(float).reindex(pd_.index).ffill()
# persistent stress and stronger-than-normal VIX, lagged observation only
active=(vc>vc.rolling(60).median())&(vc>vc.rolling(20).mean())
f=(-r10/(v*np.sqrt(10))).where(active,np.nan).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(pd_)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c):rows.append((pd_.index[i],c,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ir=lambda x:x.mean()/x.std(ddof=1)
print('universe',len(D),'dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'active',round(active.mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic;print('regime',a,b,len(z),round(z.mean(),6),round(ir(z),6))
for h in [1,3,5,10]:
 y=pd_.pct_change(h).shift(-h);rr=[]
 for i in range(len(pd_)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:rr.append(z.f.corr(z.y))
 print('decay',h,len(rr),round(np.nanmean(rr),6))
f.to_csv('scripts/miner_1_20311127_vix_stress_reversal10_signal.csv')
