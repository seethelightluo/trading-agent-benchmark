import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close']
r=pd.DataFrame(px).pct_change(); dr=dxy.pct_change()
# negative rolling beta to DXY: assets that hedge dollar strength receive high score
beta=pd.DataFrame(index=r.index,columns=U,dtype=float)
for s in U:
    cov=r[s].rolling(60,min_periods=45).cov(dr)
    var=dr.rolling(60,min_periods=45).var()
    beta[s]=-cov/var
fwd=r.shift(-1)
ics=[]; dates=[]; turnovers=[]; vals=[]
for dt in beta.index:
    x=beta.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
      if np.isfinite(ic): ics.append(ic); dates.append(dt); vals.append(len(z))
# rank turnover consecutive dates
prev=None
for dt in dates:
 q=beta.loc[dt].dropna().rank(pct=True)
 if prev is not None:
  turnovers.append(np.mean(abs(q.reindex(U)-prev.reindex(U)).dropna()))
 prev=q
A=np.array(ics); mean=A.mean(); sd=A.std(ddof=1)
print('idea=negative_60d_dxy_beta dates',len(A),'avg_n',np.mean(vals),'coverage',np.mean(vals)/15)
print('IC',mean,'ICIR',mean/sd,'hit',np.mean(A>0),'turnover',np.mean(turnovers))
for h in [5,10,20]:
 z=[]
 for dt in beta.index:
  x=beta.loc[dt]; y=r.shift(-h).loc[dt] if h==1 else (r.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt])
  q=pd.concat([x,y],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print(h,'d IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'n',len(z))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=[v for d,v in zip(dates,A) if a<=str(d.year)<=b]
 print('regime',a,b,'n',len(z),'ic',np.mean(z) if z else np.nan,'icir',np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)