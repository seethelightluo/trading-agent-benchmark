import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); ret=P/P.shift(20)-1; down=r.where(r<0,0).rolling(20,min_periods=15).std(); F=ret/(down+1e-5)
rows=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
D=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('revalidation range',D.index.min(),D.index.max(),'dates',len(D),'avgN',D.n.mean(),'coverage',D.n.mean()/15)
for a,b in [('2026-07-01','2026-12-31'),('2027-01-01','2027-06-30'),('2027-07-01','2027-12-31'),('2028-01-01','2028-01-13')]:
 x=D.loc[a:b,'ic'].dropna(); print(a,b,'n',len(x),'ic',x.mean(),'std',x.std(),'icir',x.mean()/x.std() if len(x)>1 else np.nan,'hit',(x>0).mean() if len(x) else np.nan)
for h in [1,3,5,10]:
 Y=P.shift(-h)/P-1; vals=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman'))
 x=pd.Series(vals).dropna(); print('h',h,'n',len(x),'ic',x.mean(),'icir',x.mean()/x.std())
