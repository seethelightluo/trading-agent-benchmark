import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try: d=get_stock_daily_data(s, days=6000)
 except Exception as e: print('skip',s,e); continue
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
raw=r.rolling(10).sum().shift(1)/(r.rolling(40).std().shift(1)*np.sqrt(252)+1e-8)
rows=[]
for i in range(40,len(P)-41):
 x=raw.iloc[i].dropna()
 if len(x)<8: continue
 x=x.clip(x.quantile(.05),x.quantile(.95)); x=x-x.mean()
 for h in [5,10,20,40]:
  y=(P.iloc[i+h]/P.iloc[i]-1).reindex(x.index).dropna(); xx=x.reindex(y.index)
  if len(y)>=8 and xx.std()>0 and y.std()>0: rows.append((P.index[i],h,xx.corr(y),len(y)))
D=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('universe',len(px),'dates',len(P),'observations',len(D),'avg_n',D.n.mean() if len(D) else 0)
for h,g in D.groupby('h'):
 print('H',h,'dates',len(g),'IC %.6f ICIR %.6f hit %.4f'%(g.ic.mean(),g.ic.mean()/(g.ic.std(ddof=1)+1e-12)*np.sqrt(len(g)),(g.ic>0).mean()))
print('coverage',raw.notna().sum(axis=1).mean()/15,'turnover_proxy',raw.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2025-12-31'),('2026','2030-12-31'),('2031','2035-12-31')]:
 g=D[(D.h==20)&(D.date>=a)&(D.date<=b)]; print(a,'n',len(g),'ic',g.ic.mean() if len(g) else np.nan,'icir',g.ic.mean()/(g.ic.std(ddof=1)+1e-12)*np.sqrt(len(g)) if len(g)>2 else np.nan)
print('last',P.index[-1] if len(P) else None)
