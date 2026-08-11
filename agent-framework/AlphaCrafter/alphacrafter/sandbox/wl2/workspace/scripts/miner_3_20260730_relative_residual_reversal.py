import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=D['SPX'].index; C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U})
# Relative-value shock: lagged asset return minus same-day cross-sectional median return.
# Negative residuals are expected to mean-revert; factor is minus residual magnitude.
for look in [2,3,5,8]:
 r=C.pct_change(look); residual=r.sub(r.median(axis=1),axis=0).shift(1); F=-residual
 Y=C.shift(-1).div(C)-1; q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.f,z.y).statistic
   if np.isfinite(x):q.append(x);ns.append(len(z));ds.append(dt)
 q=np.array(q)
 print('look',look,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))
 if look==3:
  for yr in range(2020,2027):
   a=q[[d.year==yr for d in ds]]
   print('regime',yr,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),5) if len(a)>1 else None)
  print('recent252',round(q[-252:].mean(),6),round(q[-252:].mean()/q[-252:].std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# decay for selected look 3
r=C.pct_change(3); F=-(r.sub(r.median(axis=1),axis=0).shift(1))
for h in [1,3,5,10]:
 Y=C.shift(-h).div(C)-1;q=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 q=np.array(q);print('decay',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('n instruments',len(U),'total dates',len(dates))
