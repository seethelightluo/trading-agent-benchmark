import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-12-02'))
dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
mom20=C/C.shift(20)-1; rev5=-(C/C.shift(5)-1)
breadth=(mom20>0).sum(axis=1)/mom20.notna().sum(axis=1)
vol=r.rolling(20,min_periods=15).std().mean(axis=1); med=vol.rolling(252,min_periods=60).median()
raw=mom20.where((breadth>=.55)&(vol<=med),rev5); F=raw.shift(1); Y=C.shift(-1).div(C)-1
for h in [1,3,5,10]:
 y=C.shift(-h).div(C)-1; vals=[]; used=[]; ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v): vals.append(v);used.append(dt);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
  z=a[[lo<=d.year<=hi for d in used]]; print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
 if h==1:
  for n in [63,126,252,504]:
   z=a[-n:]; print('recent',n,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date(),'universe',len(U))
print('artifact formula: lag1(if breadth20>=0.55 and mean cross-asset vol20 <= trailing252 median, return20, -return5)')
