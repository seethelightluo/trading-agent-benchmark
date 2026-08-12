import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,3000)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std(); r5=p.pct_change(5)
vx=pd.read_csv('../persistent/index_data/VIX.csv'); vx.date=pd.to_datetime(vx.date)
v=vx.set_index('date').close.astype(float).reindex(p.index).ffill()
stress=(v>v.rolling(60).median())&(v>v.rolling(20).mean())
breadth=(r5<0).mean(axis=1)>0.55
active=stress&breadth
f=(-r5/(vol*np.sqrt(5))).where(active,np.nan).replace([np.inf,-np.inf],np.nan)
def calc(y):
 rows=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c): rows.append((p.index[i],c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); return q
q=calc(r.shift(-1)); ir=lambda x:x.mean()/x.std(ddof=1)
print('universe',len(D),'calendar',p.index.min().date(),p.index.max().date(),'dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'active',round(active.mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031'),('2030','2032')]:
 z=q.loc[a:b].ic; print('regime',a,b,len(z),round(z.mean(),6),round(ir(z),6))
for h in [1,3,5,10]:
 z=calc(p.pct_change(h).shift(-h)); print('decay',h,len(z),round(z.ic.mean(),6))
f.to_csv('scripts/miner_1_20320108_vix_breadth_reversal5_signal.csv')
