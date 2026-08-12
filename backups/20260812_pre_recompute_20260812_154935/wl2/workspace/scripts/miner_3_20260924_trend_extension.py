import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-09-23'));dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});r=C.pct_change()
# Continuation after a stable 20d trend, penalized by a 5d extension; lagged one day.
trend=r.rolling(20,min_periods=15).sum(); ext=r.rolling(5,min_periods=4).sum()
F=(trend.rank(axis=1,pct=True)-.5*ext.rank(axis=1,pct=True)).rank(axis=1,pct=True).shift(1);Y=r.shift(-1)
q=[];ns=[];ds=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
q=np.array(q);defm=lambda x:(len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()))
print('idea trend-extension interaction; end',end.date(),'universe',len(U));print('dates avgN IC ICIR hit',len(q),round(np.mean(ns),2),*[round(x,6) for x in defm(q)[1:]],'coverage',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:print('recent',k,defm(q[-k:]))
for h in [3,5]:
 yy=C.shift(-h).div(C)-1;qq=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:qq.append(spearmanr(z.f,z.y).statistic)
 print('decay',h,defm(np.array(qq)))
print('signal_artifact','formula=rank_cs(rolling20_return)-0.5*rank_cs(rolling5_return), outer rank, lag=1')
