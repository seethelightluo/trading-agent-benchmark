import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Candidate: short-horizon volatility-scaled continuation. Recent 10d return is
# normalized by 40d volatility, rewarding persistent moves while avoiding raw-risk bias.
F=(P/P.shift(10)-1)/(r.rolling(40,min_periods=25).std()+1e-8)
rows=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman'),z.f.corr(z.y,method='pearson')))
D=pd.DataFrame(rows,columns=['date','n','sic','pic']).set_index('date')
print('range',D.index.min(),D.index.max(),'dates',len(D),'avgN',D.n.mean(),'coverage',D.n.mean()/15)
for c in ['sic','pic']:
 x=D[c].dropna();print(c,'IC',x.mean(),'std',x.std(),'ICIR',x.mean()/x.std(),'hit',(x>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2028')]:
 x=D.loc[a:b,'sic'].dropna();print(a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std() if len(x)>1 else np.nan)
for h in [3,5,10]:
 Y=P.shift(-h)/P-1; vals=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:vals.append(z.f.corr(z.y,method='spearman'))
 x=pd.Series(vals).dropna();print('h',h,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std())
rank=F.rank(axis=1,pct=True);print('turnover_proxy',((rank-rank.shift(1)).abs().mean(axis=1)).dropna().mean())
