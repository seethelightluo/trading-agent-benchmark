import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=pd.Index(sorted(set.intersection(*[set(x.index) for x in D.values()])))
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
rev=-P.pct_change(3); disp=R.std(axis=1).rolling(5,min_periods=3).mean(); state=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
F=(rev.mul(state,axis=0)).shift(1); Y=P.shift(-1).div(P).sub(1);q=[];ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
q=np.asarray(q);print('idea high_dispersion_reversal','universe',15,'dates',len(dates),'ICdates',len(q),'avgN',round(np.mean(ns),2));print('IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4));print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for yr in sorted(set(dates.year)):
 v=[]
 for dt in dates[dates.year==yr]:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(v),'IC',round(np.mean(v),6),'ICIR',round(np.mean(v)/np.std(v,ddof=1),6) if len(v)>1 else None)
