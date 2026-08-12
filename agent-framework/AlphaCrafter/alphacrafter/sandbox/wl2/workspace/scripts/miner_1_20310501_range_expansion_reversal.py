import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4100)
 if x is not None: D[s]=x.set_index('date')[['open','close','high','low']]
p={k:pd.DataFrame({s:D[s][k] for s in D}).sort_index().ffill() for k in ['open','close','high','low']}
r=p['close'].pct_change(); vol=r.rolling(40,min_periods=20).std()
# Range expansion reversal: fade today's true-range surprise, normalized by trailing volatility.
prev=p['close'].shift(1); tr=pd.concat([p['high']-p['low'],(p['high']-prev).abs(),(p['low']-prev).abs()],axis=0).groupby(level=0).max()
# groupby max above preserves columns
atr=tr.rolling(20,min_periods=10).mean(); base=atr.rolling(60,min_periods=30).mean(); surprise=atr/base-1
# directional component is close location in daily range, so large directional range is faded
clv=(p['close']-p['low'])/(p['high']-p['low']).replace(0,np.nan)-0.5
f=-(surprise*clv)/(vol*np.sqrt(252))
frs={h:r.shift(-h) for h in [1,3,5,10]}
for h,y in frs.items():
 rows=[]
 for i in range(len(f)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((f.index[i],z.f.corr(z.y),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avgN',round(q.n.mean(),3),'IC %.6f ICIR %.6f hit %.3f'%(ic,ir,(q.ic>0).mean()))
 if h==1:
  for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
   z=q.loc[a:b].ic; print(a,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.to_csv('scripts/miner_1_20310501_range_expansion_reversal_signal.csv')
