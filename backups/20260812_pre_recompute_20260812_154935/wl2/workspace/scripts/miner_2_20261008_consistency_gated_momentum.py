import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-10-07'))
dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Risk-adjusted medium momentum gated by directional consistency: 20d return / 20d vol, multiplied by fraction of positive daily returns.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
raw=r.rolling(20,min_periods=15).sum()/(vol+0.005)
cons=(r.gt(0).rolling(20,min_periods=15).mean())
F=(raw*cons).rank(axis=1,pct=True).shift(1)
Y=C.shift(-10).div(C)-1
q=[]; ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
q=np.array(q)
def met(x): return (len(x),float(np.nanmean(x)),float(np.nanmean(x)/np.nanstd(x,ddof=1)),float(np.mean(x>0)))
print('idea consistency-gated risk-adjusted momentum; end',end.date(),'universe',len(U))
print('dates avgN IC ICIR hit',len(q),round(np.mean(ns),2),*[round(x,6) for x in met(q)[1:]],'coverage',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]: print('recent',k,met(q[-k:]))
for h in [3,5,10,20]:
 yy=C.shift(-h).div(C)-1; qq=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:qq.append(spearmanr(z.f,z.y).statistic)
 print('decay',h,met(np.array(qq)))
print('signal_artifact','formula=rank_cs((rolling20_return)/(rolling20_std*sqrt20+0.005)*rolling20_positive_fraction), lag=1')
