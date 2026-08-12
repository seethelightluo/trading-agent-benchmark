import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); fr=r.shift(-1)
for look in [40,60,90]:
 hi=p.rolling(look,min_periods=max(20,look//2)).max(); lo=p.rolling(look,min_periods=max(20,look//2)).min(); loc=(p-lo)/(hi-lo)
 vol=r.rolling(20,min_periods=10).std(); f=-(loc-.5)/(vol*np.sqrt(252))
 rows=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i].rename('f'),fr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print('look',look,'dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.3f'%(ic,ir,(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
  z=q.loc[a:b].ic; print(a,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
 if look==60:
  f.to_csv('scripts/miner_3_20310403_range_location_reversal_signal.csv'); print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
