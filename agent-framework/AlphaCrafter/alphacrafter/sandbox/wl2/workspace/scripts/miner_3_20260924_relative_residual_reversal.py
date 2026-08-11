import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-09-23'))
dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change(3)
# Lagged relative shock: asset 3d return minus cross-sectional median; negative shock should mean-revert.
F=-(r.sub(r.median(axis=1),axis=0)).shift(1); Y=C.pct_change().shift(-1)
q=[];ns=[];used=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.f,z.y).statistic
  if np.isfinite(v):q.append(v);ns.append(len(z));used.append(dt)
q=np.array(q)
def met(x): return (len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean())) if len(x)>1 else (len(x),np.nan,np.nan,np.nan)
print('idea lagged 3d cross-sectional residual reversal; end',end.date(),'universe',len(U))
print('dates avgN IC ICIR hit coverage',len(q),round(np.mean(ns),2),*[round(x,6) for x in met(q)[1:]],round(F.notna().sum().sum()/F.size,4))
for k in [63,126,252,504]: print('recent',k,met(q[-k:]))
for a,b in [('2020','2021'),('2022','2023'),('2024','2026')]:
 x=q[[str(d.year) in ([a,b] if a!='2024' else ['2024','2025','2026']) for d in used]];print('regime',a+'-'+b,met(x))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [1,3,5,10]:
 Y=C.shift(-h).div(C)-1; qq=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: qq.append(spearmanr(z.f,z.y).statistic)
 print('decay',h,met(np.array(qq)))
print('signal_artifact','formula=-(asset_3d_return-cross_sectional_median_3d_return); lag=1')
