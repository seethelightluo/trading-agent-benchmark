import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(symbol=s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').drop_duplicates('date').set_index('date');F[s]=d
# medium momentum residualized against contemporaneous cross-sectional median: persistent trend relative to peers
P=pd.concat({s:d.close.pct_change(20) for s,d in F.items()},axis=1)
M=P.median(axis=1)
rows=[]
for s,d in F.items():
 sig=(P[s]-M).shift(1)
 for h in [1,3,5]:
  f=d.close.shift(-h)/d.close-1
  z=pd.DataFrame({'sig':sig,'f':f}).dropna()
  for dt,r in z.iterrows(): rows.append((dt,s,h,r.sig,r.f))
x=pd.DataFrame(rows,columns=['date','symbol','h','sig','f'])
for h in [1,3,5]:
 y=x[x.h==h]; vals=[]
 for dt,g in y.groupby('date'):
  if len(g)>=8: vals.append(g.sig.corr(g.f,method='spearman'))
 z=pd.Series(vals).dropna();print('h',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027')]:
  # dates absent, align by index unavailable; recompute
  pass
print('coverage',x.groupby('date').symbol.nunique().mean()/15)
x.to_csv('scripts/miner_2_20300919_peer_relative_signal.csv',index=False)
