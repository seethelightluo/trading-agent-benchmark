import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,1800) for s in U}
fac={}; rets={}
for s,d in px.items():
 if d is None or len(d)<30: continue
 x=d.copy(); x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); r=x.close.pct_change()
 fac[s]=(r.rolling(10,min_periods=8).sum()/r.abs().rolling(10,min_periods=8).sum()).shift(1)
 rets[s]=r
F=pd.DataFrame(fac); R=pd.DataFrame(rets); rows=[]
for dt in F.index:
 vals=[]
 for s in U:
  if s in F and s in R:
   ix=R[s].index; pos=ix.searchsorted(dt,side='right')
   if pos<len(ix) and pd.notna(F.at[dt,s]) and pd.notna(R[s].iloc[pos]): vals.append((F.at[dt,s],R[s].iloc[pos]))
 if len(vals)>=8:
  x,y=np.array(vals).T
  if np.std(x)>0 and np.std(y)>0: rows.append((dt,np.corrcoef(x,y)[0,1],len(vals)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,q in [('all',z),('2020-22',z.loc['2020':'2022']),('2023-24',z.loc['2023':'2024']),('2025-26',z.loc['2025':'2026'])]:
 ic=q.ic.dropna(); print(label,'dates',len(ic),'avgN',round(q.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
print('assets',len(F.columns),'coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
