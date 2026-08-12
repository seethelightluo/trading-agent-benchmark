import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-02-10'); dates=D['SPX'].index[(D['SPX'].index>='2020-05-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
vol=R.rolling(30,min_periods=20).std()*np.sqrt(252)
pos=(C-C.rolling(90,min_periods=60).min())/(C.rolling(90,min_periods=60).max()-C.rolling(90,min_periods=60).min())
cons=(R>0).rolling(30,min_periods=20).mean()
F=(C.pct_change(30)/vol*(0.5+pos)*(.5+cons)).shift(1)
for h in [1,3,5,10]:
 y=C.shift(-h).div(C)-1;a=[];ds=[];ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
  z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
print('formula: (30d return / annualized 30d volatility) * (0.5 + 90d range position) * (0.5 + 30d positive-day consistency), lagged one day')
