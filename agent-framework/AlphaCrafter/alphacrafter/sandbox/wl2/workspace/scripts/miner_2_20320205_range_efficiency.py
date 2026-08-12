import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3200)
 if d is not None: C[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index().ffill(); r=p.pct_change()
# Range-efficiency trend: signed 10d move divided by the sum of absolute daily moves.
# It rewards directional persistence while suppressing choppy trends; all inputs are lagged by one day.
eff=r.rolling(10).sum().div(r.abs().rolling(10).sum().replace(0,np.nan))
# risk-scale and cross-sectional standardization; signal is used for continuation.
f=eff.div(r.rolling(20).std()).shift(1)
def ic_series(h):
 y=p.pct_change(h).shift(-h); out=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c): out.append((p.index[i],c,len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
def ir(x): return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
q=ic_series(1)
print('universe',len(C),'dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031'),('2030','2032')]:
 z=q.loc[a:b].ic; print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z) else None)
for h in [3,5,10]:
 z=ic_series(h); print('decay',h,'dates',len(z),'IC',round(z.ic.mean(),6) if len(z) else None)
f.to_csv('scripts/miner_2_20320205_range_efficiency_signal.csv')
