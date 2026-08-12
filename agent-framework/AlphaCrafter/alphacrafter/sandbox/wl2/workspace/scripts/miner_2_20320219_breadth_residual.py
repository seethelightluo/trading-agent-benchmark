import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<100:d=get_index_daily_data(s,days=3200)
 if d is not None:C[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1); res=r.sub(m,axis=0)
breadth=(r.rolling(5).sum()<0).mean(axis=1)
f=(-(res.rolling(5).sum()).div(r.rolling(20).std())).where(breadth>=.55).shift(1)
def ir(x):return x.mean()/x.std(ddof=1) if len(x)>1 and x.std(ddof=1)>0 else np.nan
def calc(h):
 y=p.pct_change(h).shift(-h); rows=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c):rows.append((p.index[i],c,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
q=calc(1);print('universe',len(C),'dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(ir(q.ic),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'active_dates',round((breadth>=.55).mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),4))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 z=q.loc[a:b].ic;print('regime',a,b,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(ir(z),6) if len(z)>1 else None)
for h in [3,5,10]:
 z=calc(h);print('decay',h,'dates',len(z),'IC',round(z.ic.mean(),6) if len(z) else None)
f.to_csv('scripts/miner_2_20320219_breadth_residual_signal.csv')
