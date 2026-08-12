import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); R20=P/P.shift(20)-1
breadth=(R20>0).mean(axis=1)
F=R20.copy(); F.loc[breadth<.5]=-R20.loc[breadth<.5]
rows=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
D=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); print('range',D.index.min(),D.index.max(),'dates',len(D),'avgN',D.n.mean(),'coverage',D.n.mean()/15)
for a in ['2026','2027','2028']:
 x=D.loc[a,'ic'].dropna(); print(a,len(x),x.mean(),x.mean()/x.std() if len(x)>1 else np.nan,(x>0).mean())
x=D.ic.dropna(); print('daily',x.mean(),x.std(),x.mean()/x.std(),(x>0).mean())
for h in [3,5,10]:
 Y=P.shift(-h)/P-1; vals=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman'))
 q=pd.Series(vals).dropna(); print('h',h,len(q),q.mean(),q.mean()/q.std())
rank=F.rank(axis=1,pct=True); print('turnover',((rank-rank.shift()).abs().mean(axis=1)).dropna().mean())
