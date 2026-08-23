import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-10-21')
F={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,3000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date').set_index('date'); d['r']=d.close.pct_change();F[s]=d
# Range-normalized medium momentum: return over 10 sessions scaled by average true range,
# with efficiency (absolute displacement / path) to favor persistent trends.
rows=[]
for s,d in F.items():
 tr=(d.high-d.low)/d.close
 path=d.r.abs().rolling(20).sum(); eff=d.close.pct_change(10).abs()/path
 f=d.close.pct_change(10)/(tr.rolling(20).mean()*np.sqrt(10))* (0.5+eff)
 fr=d.close.pct_change().shift(-1)
 for dt in d.index:
  if pd.notna(f.get(dt)) and pd.notna(fr.get(dt)):rows.append((dt,s,f.loc[dt],fr.loc[dt]))
x=pd.DataFrame(rows,columns=['date','sym','f','r']); n=x.groupby('date').size(); ic=x.groupby('date').apply(lambda z:z.f.corr(z.r),include_groups=False).dropna()
print('range_normalized_persistent_momentum');print('dates',len(ic),'avg_names',n.mean(),'coverage',n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for h in [5,10,20]:
 z=[]
 for s,d in F.items():
  tr=((d.high-d.low)/d.close).rolling(20).mean(); eff=d.close.pct_change(10).abs()/d.r.abs().rolling(20).sum(); f=d.close.pct_change(10)/(tr*np.sqrt(10))*(.5+eff); r=d.close.pct_change(h).shift(-h); q=pd.DataFrame({'f':f,'r':r}).dropna();z.extend([(dt,q.loc[dt].f,q.loc[dt].r) for dt in q.index])
 a=pd.DataFrame(z,columns=['dt','f','r']).groupby('dt').apply(lambda q:q.f.corr(q.r),include_groups=False).dropna();print('h',h,'IC %.7f ICIR %.7f'%(a.mean(),a.mean()/a.std()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ic.loc[a:b];print(a,b,len(q),q.mean(),q.mean()/q.std())
print('turnover proxy',x.sort_values(['sym','date']).groupby('sym').f.apply(lambda z:z.rank(pct=True).diff().abs().mean()).mean())
