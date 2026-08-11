import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}; dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<='2026-12-16')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change(); M=r.mean(axis=1); vr=M.rolling(60,min_periods=45).var(); b=pd.DataFrame({s:r[s].rolling(60,min_periods=45).cov(M)/(vr+1e-8) for s in U}); m30=(1+M).rolling(30,min_periods=25).apply(np.prod,raw=True)-1
# Residual momentum multiplied by 30d fraction of positive asset returns; lag one day.
res=C.pct_change(30)-b.mul(m30,axis=0); cons=r.gt(0).rolling(30,min_periods=20).mean(); F=res.mul(cons).rank(axis=1,pct=True).shift(1)
def met(x): x=np.array(x); return len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0)
for h in [1,5,10]:
 Y=C.shift(-h).div(C)-1;q=[];ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 print(h,len(q),round(np.mean(ns),2),*[round(v,6) for v in met(q)[1:]])
print('coverage',F.notna().sum().sum()/F.size,'turn',F.diff().abs().mean(axis=1).mean())
print('signal_artifact formula=rank_cs((ret30-beta60_eqw*ret30_eqw)*positive_fraction30), lag=1')
