import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-12-16')); dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U}); V=pd.DataFrame({s:D[s].volume.reindex(dates) for s in U})
clv=(2*C-H-L)/(H-L).replace(0,np.nan); volsur=V/V.rolling(20,min_periods=10).median()-1
# Smoothed 10-day pressure, with volume confirmation capped to avoid crypto volume outliers.
F=(clv.rolling(10,min_periods=6).mean()*(1+volsur.clip(-.5,1.0))).shift(1); Y=C.shift(-1).div(C)-1
for h in [1,3,5,10]:
 y=C.shift(-h).div(C)-1; q=[]; used=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v):q.append(v);used.append(dt);ns.append(len(z))
 a=np.array(q); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
   z=a[[lo<=d.year<=hi for d in used]]; print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
  for n in [63,126,252,504]:
   z=a[-n:]; print('recent',n,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date(),'universe',len(U))
print('artifact formula: lag1(rolling10mean(CLV)*(1+clip(volume/rolling20median(volume)-1,-.5,1)))')
