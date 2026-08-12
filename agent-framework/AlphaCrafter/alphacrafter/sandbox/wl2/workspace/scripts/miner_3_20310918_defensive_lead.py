import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; DEF=['XAU','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();
# Relative defensive-lead signal: asset 10d return relative to defensive basket,
# volatility scaled and activated when defensive basket outperforms risk assets.
dret=r[DEF].mean(axis=1); risk=[x for x in U if x not in DEF]; stress=(dret.rolling(10).sum()-r[risk].mean(axis=1).rolling(10).sum())>0
f=r.rolling(10).sum().sub(dret.rolling(10).sum(),axis=0).div(r.rolling(20).std().replace(0,np.nan)); f=f.where(stress,np.nan)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  c=z.f.corr(z.y)
  if np.isfinite(c):rows.append((p.index[i],len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'active',round(stress.mean(),4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-31',a.index>='2026-01-01')]:
 z=a.loc[mask].ic;print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
for h in [3,5,10]:
 yy=p.pct_change(h).shift(-h)/h;rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),yy.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c):rr.append(c)
 print('decay',h,len(rr),round(np.mean(rr),6))
f.to_csv('scripts/miner_3_20310918_defensive_lead_signal.csv');print('signal_rows',int(f.notna().sum().sum()))
