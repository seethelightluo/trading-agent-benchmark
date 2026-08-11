import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Overnight gap reversal: fade today's open-to-prior-close gap, known at prior close? Production at decision after completed day uses prior day's gap.
# signal on date t is prior day's gap, forward return is t+1 close / t close.
G=pd.DataFrame({s:(x.open/x.close.shift(1)-1).shift(1) for s,x in D.items()}).sort_index()
Y=pd.DataFrame({s:x.close.shift(-1)/x.close-1 for s,x in D.items()}).sort_index()
ics=[]; ns=[]
for dt in G.index:
 z=pd.DataFrame({'f':-G.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
a=np.asarray(ics); print('dates',len(a),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in [5,10]:
 Yh=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index(); q=[]
 for dt in G.index:
  z=pd.DataFrame({'f':-G.loc[dt],'y':Yh.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 q=np.asarray(q);print('horizon',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for y in range(2020,2027):
 q=[]
 for dt in G.loc[str(y)].index:
  z=pd.DataFrame({'f':-G.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 print('regime',y,'dates',len(q),'IC',round(np.mean(q),6) if q else None)
print('turnover',G.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'valid_cell_coverage',G.notna().sum().sum()/G.size)
# correlation against existing signal artifacts approximated by pooled ranks
for name,F in [('reversal5',-pd.DataFrame({s:D[s].close.pct_change(5) for s in U})),('momentum20',pd.DataFrame({s:D[s].close.pct_change(20) for s in U}))]:
 z=pd.concat([(-G).stack().rename('new'),F.stack().rename('old')],axis=1).dropna();print('corr',name,z.corr().iloc[0,1])
