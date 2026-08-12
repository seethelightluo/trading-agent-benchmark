import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=D['SPX'].index
O=pd.DataFrame({s:D[s].open.reindex(dates) for s in U}); C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U})
# Lagged intraday reversal: yesterday's close-open move, demeaned across assets.
for k in [1,2,3]:
 raw=(C/O-1).rolling(k).sum(); F=-(raw.sub(raw.median(axis=1),axis=0)).shift(1); Y=C.shift(-1).div(C)-1;q=[];ns=[];ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   a=spearmanr(z.f,z.y).statistic
   if np.isfinite(a):q.append(a);ns.append(len(z));ds.append(dt)
 q=np.array(q);print('k',k,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))
 if k==1:
  for yr in range(2020,2027):
   a=q[[d.year==yr for d in ds]];print('regime',yr,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),5) if len(a)>1 else None)
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'instruments',len(U),'total dates',len(dates))
