import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=pd.Index(sorted(set.intersection(*[set(x.index) for x in D.values()])))
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change(); v=R.rolling(20,min_periods=15).std()*np.sqrt(20)
a=P.pct_change(3)/v; b=P.pct_change(10)/v
F=(a.rank(axis=1,pct=True)*.5+b.rank(axis=1,pct=True)*.5).shift(1)
print('idea rank_blend_volnorm_3_10; universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1);q=[];ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q);print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for yr in sorted(set(dates.year)):
 vals=[]
 for dt in dates[dates.year==yr]:
  y=P.shift(-1).div(P).sub(1);z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:vals.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(vals),'IC',round(np.mean(vals),6),'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),6) if len(vals)>1 else None)
