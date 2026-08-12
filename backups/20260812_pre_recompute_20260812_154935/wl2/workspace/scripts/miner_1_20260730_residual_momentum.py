import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U}); m=R['SPX']
# Residual momentum: medium return less rolling SPX beta times SPX return; lagged one day.
mu=m.rolling(60,min_periods=45).mean(); var=((m-mu)**2).rolling(60,min_periods=45).mean()
F=pd.DataFrame(index=dates)
for s in U:
 x=R[s]; xu=x.rolling(60,min_periods=45).mean(); cov=((x-xu)*(m-mu)).rolling(60,min_periods=45).mean(); beta=cov/var
 F[s]=(R[s].rolling(20,min_periods=15).sum()-beta*R['SPX'].rolling(20,min_periods=15).sum()).shift(1)
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].close.shift(-h).div(D[s].close).sub(1).reindex(dates) for s in U}); q=[];ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q);print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=[]
   for dt in F.loc[str(yr)].index:
    z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
    if len(z)>=8:x.append(spearmanr(z.f,z.y).statistic)
   print('regime',yr,'dates',len(x),'IC',round(np.mean(x),6) if x else None,'ICIR',round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
