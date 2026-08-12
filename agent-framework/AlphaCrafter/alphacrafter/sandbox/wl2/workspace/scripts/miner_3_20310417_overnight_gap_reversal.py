import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=4100)
 if x is not None:D[s]=x.set_index('date')[['open','close']]
p={k:pd.DataFrame({s:D[s][k] for s in D}).sort_index().ffill() for k in ['open','close']}
r=p['close'].pct_change(); fr=r.shift(-1); gap=p['open']/p['close'].shift(1)-1; vol=r.rolling(20,min_periods=10).std()
for w in [1,3,5]:
 f=-gap.rolling(w,min_periods=w).mean()/(vol*np.sqrt(252)); rows=[]
 for i in range(len(f)-1):
  z=pd.concat([f.iloc[i].rename('f'),fr.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:rows.append((f.index[i],z.f.corr(z.y),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean();ir=ic/q.ic.std(ddof=1)
 print('window',w,'dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.3f'%(ic,ir,(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
  z=q.loc[a:b].ic;print(a,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
 if w==1:
  f.to_csv('scripts/miner_3_20310417_overnight_gap_signal.csv');print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
