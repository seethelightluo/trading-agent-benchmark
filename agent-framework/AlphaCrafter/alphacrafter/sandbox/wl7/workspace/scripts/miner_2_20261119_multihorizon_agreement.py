import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-18'); F={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date').set_index('date');d['r']=d.close.pct_change();F[s]=d
# Candidate: multi-horizon trend agreement, using lagged 5/20/60d returns and volatility penalty.
rows=[]
for s,d in F.items():
 vol=d.r.rolling(20).std(); f=(d.close.pct_change(5)/vol + d.close.pct_change(20)/vol + d.close.pct_change(60)/vol)/3
 # forward one-day return, all inputs are available at dt close
 r=d.r.shift(-1)
 for dt in d.index:
  if pd.notna(f.get(dt)) and pd.notna(r.get(dt)): rows.append((dt,s,f.loc[dt],r.loc[dt]))
x=pd.DataFrame(rows,columns=['date','sym','f','r']); n=x.groupby('date').size(); ic=x.groupby('date').apply(lambda z:z.f.corr(z.r),include_groups=False).dropna()
print('multi_horizon_agreement');print('dates',len(ic),'avg_names',round(n.mean(),3),'coverage',round(n.mean()/15,4));print('IC %.8f ICIR %.8f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for h in [5,10,20]:
 z=[]
 for s,d in F.items():
  vol=d.r.rolling(20).std();f=(d.close.pct_change(5)/vol+d.close.pct_change(20)/vol+d.close.pct_change(60)/vol)/3;r=d.close.pct_change(h).shift(-h);q=pd.DataFrame({'f':f,'r':r}).dropna();z.extend([(dt,q.loc[dt].f,q.loc[dt].r) for dt in q.index])
 a=pd.DataFrame(z,columns=['dt','f','r']).groupby('dt').apply(lambda q:q.f.corr(q.r),include_groups=False).dropna();print('h',h,'IC %.7f ICIR %.7f'%(a.mean(),a.mean()/a.std()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ic.loc[a:b];print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
print('turnover_proxy',x.sort_values(['sym','date']).groupby('sym').f.apply(lambda z:z.rank(pct=True).diff().abs().mean()).mean())
