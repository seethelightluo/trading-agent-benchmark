import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-10-21'))
dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Adaptive trend/reversal: stable positive breadth favors 20d trend; stressed breadth favors 5d reversal.
mom20=C/C.shift(20)-1; rev5=-(C/C.shift(5)-1)
breadth=(mom20>0).sum(axis=1)/mom20.notna().sum(axis=1)
vol=r.rolling(20,min_periods=15).std().mean(axis=1)
# use only information through t-1; factor is lagged
raw=pd.DataFrame(index=dates,columns=U,dtype=float)
for dt in dates:
    if breadth.loc[dt] >= .55 and vol.loc[dt] <= vol.rolling(252,min_periods=60).median().loc[dt]: raw.loc[dt]=mom20.loc[dt]
    else: raw.loc[dt]=rev5.loc[dt]
F=raw.shift(1); Y=C.shift(-1).div(C)-1
q=[]; ns=[]; used=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.f,z.y).statistic
  if np.isfinite(v): q.append(v);ns.append(len(z));used.append(dt)
q=np.array(q)
def met(x): return (len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()))
print('idea adaptive trend/reversal; end',end.date(),'universe',len(U))
print('dates avgN IC ICIR hit coverage turnover',len(q),round(np.mean(ns),2),*[round(x,6) for x in met(q)[1:]],round(F.notna().sum().sum()/F.size,4),round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]: print('recent',k,met(q[-k:]))
for a,b in [('2020','2021'),('2022','2023'),('2024','2026')]:
 print('regime',a+'-'+b,met(q[[int(a)<=d.year<=int(b) for d in used]]))
for h in [3,5,10]:
 yy=C.shift(-h).div(C)-1; qq=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: qq.append(spearmanr(z.f,z.y).statistic)
 print('decay',h,met(np.array(qq)))
print('signal_artifact formula=lag1(if breadth20>=.55 and mean cross-asset vol20 below trailing252 median, return20, -return5)')
