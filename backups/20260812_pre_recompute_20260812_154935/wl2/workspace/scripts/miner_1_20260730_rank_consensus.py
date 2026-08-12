import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index; C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U})
# Multi-horizon rank consensus, with all observations lagged one session.
rs=[]
for n in [5,20,60]: rs.append(C.pct_change(n).shift(1).rank(axis=1,pct=True))
F=sum(rs)/len(rs)
for h in [1,5,10]:
 Y=C.shift(-h)/C-1;q=[];ns=[];ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.array(q);print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.diff().abs().mean(axis=1).mean(),4))
  for yr in range(2020,2027):
   x=np.array([q[i] for i,d in enumerate(ds) if d.year==yr])
   if len(x):print('regime',yr,len(x),round(x.mean(),6))
