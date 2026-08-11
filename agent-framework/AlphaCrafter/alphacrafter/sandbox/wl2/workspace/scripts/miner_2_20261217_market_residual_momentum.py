import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2026-12-16'); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change(); M=r.mean(axis=1)
vr=M.rolling(60,min_periods=45).var(); beta=pd.DataFrame({s:r[s].rolling(60,min_periods=45).cov(M)/(vr+1e-8) for s in U})
R30=C.pct_change(30); m30=(1+M).rolling(30,min_periods=25).apply(np.prod,raw=True)-1
F=(R30-beta.mul(m30,axis=0)).rank(axis=1,pct=True).shift(1)
def met(x):
 x=np.array(x); return (len(x),float(np.nanmean(x)),float(np.nanmean(x)/np.nanstd(x,ddof=1)),float(np.mean(x>0)))
print('idea market-residual momentum; end',end.date(),'universe',len(U))
for h in [1,5,10,20]:
 Y=C.shift(-h).div(C)-1;q=[];ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 print('horizon',h,'dates avgN IC ICIR hit',len(q),round(np.mean(ns),2),*[round(x,6) for x in met(q)[1:]])
q=[]; Y=C.shift(-10).div(C)-1
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
for k in [63,252,504]: print('recent10',k,met(q[-k:]))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.diff().abs().mean(axis=1).mean(),5))
print('signal_artifact formula=rank_cs(ret30-beta60_eqw*ret30_eqw), lag=1')
