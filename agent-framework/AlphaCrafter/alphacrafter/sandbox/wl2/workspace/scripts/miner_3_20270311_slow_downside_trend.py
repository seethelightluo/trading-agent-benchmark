import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-03-10'); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
# Slow trend quality: 60d return, penalized by downside deviation and recent return dispersion.
down=R.where(R<0,0).pow(2).rolling(60,min_periods=40).mean().pow(.5)
F=(C.pct_change(60)/down.replace(0,np.nan)).shift(1)
y=C.shift(-1).div(C)-1
A=[];ds=[];ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.f,z.y).statistic
  if np.isfinite(q):A.append(q);ds.append(dt);ns.append(len(z))
a=np.array(A);print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
