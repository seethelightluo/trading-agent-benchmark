import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dates=D['SPX'].index[(D['SPX'].index>='2025-01-01')&(D['SPX'].index<='2026-08-12')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U})
clv=-(2*(C-L)/(H-L).replace(0,np.nan))
for label,F in [('clv15',clv.rank(axis=1,pct=True).add(.15*C.pct_change(20).rank(axis=1,pct=True))).items():
 F=F.shift(1); Y=C.pct_change().shift(-1); q=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q); print(label,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))
 for k in [63,126,252]:
  x=q[-k:]; print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
 print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
