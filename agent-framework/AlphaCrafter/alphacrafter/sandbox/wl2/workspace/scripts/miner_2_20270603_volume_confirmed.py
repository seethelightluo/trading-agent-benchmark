import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-06-02'); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); V=pd.DataFrame({s:D[s].volume.reindex(dates) for s in U}); R=C.pct_change()
trend=C.pct_change(20); rv=R.rolling(20,min_periods=15).std(); vr=V/V.rolling(60,min_periods=30).median(); confirm=vr.clip(0.5,2.0).rank(axis=1,pct=True)
F=(trend/rv.replace(0,np.nan)*(0.5+0.5*confirm)).shift(1); y=C.shift(-1).div(C)-1
ics=[];ds=[];ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.f,z.y).statistic
  if np.isfinite(q):ics.append(q);ds.append(dt);ns.append(len(z))
a=np.array(ics);print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4));print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for h in [3,5,10]:
 yy=C.shift(-h).div(C)-1;aa=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.f,z.y).statistic)
 aa=np.array(aa);print('h',h,'n',len(aa),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6))
