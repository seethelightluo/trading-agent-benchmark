import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-02-24'); dates=D['SPX'].index[(D['SPX'].index>='2020-02-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Directional persistence efficiency: cumulative 20d return, scaled by volatility and
# consistency of positive days. Every input is lagged one completed day.
ret=C.pct_change(20); vol=r.rolling(20).std(); consistency=(r>0).rolling(20).mean()
F=(ret/vol*consistency).shift(1); y=C.shift(-1).div(C)-1
a=[];ds=[];ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.f,z.y).statistic
  if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
a=np.array(a); print('factor persistence_efficiency_20d','dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
for h in [3,5,10]:
 yy=C.shift(-h).div(C)-1; aa=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.f,z.y).statistic)
 aa=np.array(aa); print('H',h,'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'n',len(aa))
