import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,3000)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); v20=r.rolling(20).std(); v60=r.rolling(60).std()
# Stable low-volatility: inverse short volatility, scaled by longer volatility to avoid
# selecting temporarily quiet assets with structurally extreme risk.
f=(v60/v20).replace([np.inf,-np.inf],np.nan).rank(axis=1,pct=True)-0.5

def calc(y):
 out=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c): out.append((p.index[i],c,len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
q=calc(r.shift(-1)); ir=lambda x:x.mean()/x.std(ddof=1)
print('universe',len(D),'calendar',p.index.min().date(),p.index.max().date(),'dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 z=q.loc[a:b].ic; print('regime',a,b,len(z),round(z.mean(),6),round(ir(z),6))
for h in [1,3,5,10]:
 z=calc(p.pct_change(h).shift(-h)); print('decay',h,len(z),round(z.ic.mean(),6),round(ir(z.ic),6))
f.to_csv('scripts/miner_1_20320122_stable_lowvol_signal.csv')
