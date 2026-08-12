import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
M=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end]
dates=D['SPX'].index; C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change(); v=M.close.reindex(dates).ffill(); vr=v.pct_change()
# Market-wide VIX residual: subtract the cross-sectional median exposure response (beta estimated per asset).
vv=vr.rolling(60,min_periods=30).var(); beta=pd.DataFrame(index=dates,columns=U,dtype=float)
for s in U: beta[s]=R[s].rolling(60,min_periods=30).cov(vr)/vv
res=C.pct_change(10)-beta.mul(v.pct_change(10),axis=0)
# signal only needs lagged completed data; forward return aligned by date
F=res.shift(1); Y=C.shift(-1).div(C)-1
q=[];ns=[];ds=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  a=spearmanr(z.f,z.y).statistic
  if np.isfinite(a):q.append(a);ns.append(len(z));ds.append(dt)
q=np.array(q); print('dates',len(q),'avgN',round(np.mean(ns),2) if ns else 0,'IC',round(q.mean(),6) if len(q) else 'nan','ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else 'nan','hit',round((q>0).mean(),4) if len(q) else 'nan','coverage',round(F.notna().sum().sum()/F.size,4))
for yr in range(2020,2027):
 a=q[[d.year==yr for d in ds]]; print('regime',yr,len(a),round(a.mean(),6) if len(a) else 'nan',round(a.mean()/a.std(ddof=1),5) if len(a)>1 else None)
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'instruments',len(U),'total dates',len(dates))
