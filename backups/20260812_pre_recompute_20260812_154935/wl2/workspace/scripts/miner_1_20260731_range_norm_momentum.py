import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
# Range-normalized short momentum: recent return scaled by trailing true range, lagged.
TR=pd.DataFrame({s:(D[s].high-D[s].low).div(D[s].close).reindex(dates) for s in U})
F=R.rolling(5,min_periods=5).sum().div(TR.rolling(20,min_periods=15).mean()).shift(1)
for h in [1,5,10]:
 Y=pd.DataFrame({s:C[s].shift(-h).div(C[s]).sub(1) for s in U}); q=[]; ds=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ds.append(dt);ns.append(len(z))
 q=np.array(q); print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
  for yr in range(2020,2027):
   x=q[[d.year==yr for d in ds]]; print('regime',yr,'n',len(x),'IC',round(x.mean(),6) if len(x) else None)
 print()
