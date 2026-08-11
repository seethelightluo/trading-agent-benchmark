import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-07-14'); dates=D['SPX'].index[(D['SPX'].index>='2020-02-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); V=pd.DataFrame({s:D[s].volume.reindex(dates) for s in U}); R=C.pct_change()
# Volume-confirmed momentum: 10-day return weighted by relative volume, with cross-asset
# comparable volume z-score. Lagged one completed session.
rv=V/V.rolling(20,min_periods=10).mean(); F=(C.pct_change(10)*rv.clip(0.5,2.0)).shift(1); y=C.shift(-1).div(C)-1
A=[];ds=[];ns=[]
for dt in dates:
 q=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(q)>=8:
  ic=spearmanr(q.f,q.y).statistic
  if np.isfinite(ic): A.append(ic);ds.append(dt);ns.append(len(q))
a=np.array(A);print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 q=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [3,5,10]:
 yy=C.shift(-h).div(C)-1;aa=[]
 for dt in dates:
  q=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8:aa.append(spearmanr(q.f,q.y).statistic)
 aa=np.array(aa);print('horizon',h,'dates',len(aa),'IC',round(np.nanmean(aa),6),'ICIR',round(np.nanmean(aa)/np.nanstd(aa,ddof=1),6))
print('signal_artifact',F.tail(1).to_json())
