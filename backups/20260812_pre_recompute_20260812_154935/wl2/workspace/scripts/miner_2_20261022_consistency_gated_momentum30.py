import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-10-21'))
dates=D['SPX'].index[(D['SPX'].index>='2020-06-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# 30-session risk-adjusted momentum multiplied by directional consistency; lag one completed session.
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
raw=r.rolling(30,min_periods=20).sum()/(vol+0.005)
cons=r.gt(0).rolling(30,min_periods=20).mean()
F=(raw*cons).rank(axis=1,pct=True).shift(1)
def evaluate(Y):
 q=[]; ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q)
 return len(q),np.mean(ns),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1),np.mean(q>0),q
print('idea=30d consistency-gated risk-adjusted momentum; end=',end.date(),'universe=',len(U))
for h in [3,5,10,20]:
 out=evaluate(C.shift(-h).div(C)-1); print('horizon',h,'dates avgN IC ICIR hit',out[0],round(out[1],2),*[round(x,6) for x in out[2:5]])
q=evaluate(C.shift(-10).div(C)-1)[5]
for k in [63,126,252,504]:
 x=q[-k:];print('recent',k,'IC ICIR hit',*[round(v,6) for v in [np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0)]])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'avg breadth',round(F.notna().sum(axis=1).mean(),2))
print('signal_artifact=rank_cs((sum_return_30)/(std_daily_30*sqrt30+0.005)*positive_fraction_30),lag=1')
