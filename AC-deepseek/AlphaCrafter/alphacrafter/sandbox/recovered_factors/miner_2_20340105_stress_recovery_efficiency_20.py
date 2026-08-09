import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-01-04')
def rd(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}).loc['2020-01-01':]; R=P.pct_change(fill_method=None)
vol=R.rolling(20,min_periods=15).std(); dd=P/P.rolling(60,min_periods=30).max()-1
# Stress recovery efficiency: after broad cross-asset stress, favor assets that recovered most
# from their own recent drawdown, normalized by risk and residualized against trend/vol.
market5=R.mean(axis=1).rolling(5,min_periods=4).mean(); marketvol=R.mean(axis=1).rolling(20,min_periods=15).std()
stress=(market5 < -0.004) | (marketvol > marketvol.rolling(120,min_periods=60).quantile(.75))
raw=(P/P.shift(5)-1)/(vol*np.sqrt(5)+1e-12)
raw=raw.where(stress.shift(1).rolling(10,min_periods=1).max().astype(bool))
trend=P/P.shift(20)-1
# daily cross-sectional residualization
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for t in P.index:
 q=pd.concat([raw.loc[t].rename('y'),vol.loc[t].rename('v'),trend.loc[t].rename('tr'),dd.loc[t].rename('dd')],axis=1).dropna()
 if len(q)>=8:
  X=np.c_[np.ones(len(q)),q[['v','tr','dd']]]
  if np.linalg.matrix_rank(X)==X.shape[1]: F.loc[t,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
print('candidate=stress_recovery_efficiency_20');print('rows',len(P),'dates',F.notna().any(axis=1).sum(),'mean_n',F.notna().sum(1).replace(0,np.nan).mean(),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 ic=[]
 for i in range(len(P)-h):
  q=pd.concat([F.iloc[i].rename('f'),R.shift(-h).iloc[i].rename('r')],axis=1).dropna()
  if len(q)>=8: ic.append(q.f.corr(q.r,method='spearman'))
 print('h',h,'IC',round(np.nanmean(ic),6),'ICIR',round(np.nanmean(ic)/(np.nanstd(ic,ddof=1)+1e-12),6),'hit',round(np.mean(np.array(ic)>0),4),'n',len(ic))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
