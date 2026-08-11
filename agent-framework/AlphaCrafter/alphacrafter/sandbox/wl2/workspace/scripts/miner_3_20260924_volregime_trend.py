import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-09-23'));dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});r=C.pct_change(); vol=r.rolling(20,min_periods=15).std(); long=r.rolling(60,min_periods=40).std()
# Low-volatility continuation: medium trend favored, scaled by inverse relative volatility; lagged.
F=(r.rolling(20,min_periods=15).sum()/(vol/(long+1e-8)+1e-8)).shift(1);Y=r.shift(-1)
q=[];ns=[];ds=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
q=np.array(q);m=lambda x:(len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()))
print('idea volatility-regime trend; end',end.date(),'universe',len(U));print('dates avgN IC ICIR hit',len(q),round(np.mean(ns),2),*[round(x,6) for x in m(q)[1:]],'coverage',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:print('recent',k,m(q[-k:]))
for h in [3,5]:
 yy=C.shift(-h).div(C)-1;qq=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:qq.append(spearmanr(z.f,z.y).statistic)
 print('decay',h,m(np.array(qq)))
print('signal_artifact','formula=20d_return/(20d_vol/60d_vol), lag=1')
