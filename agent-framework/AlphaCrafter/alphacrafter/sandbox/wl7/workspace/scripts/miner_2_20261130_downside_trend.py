import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
p=pd.DataFrame(px).sort_index().loc[:'2026-11-30']
r=p.pct_change()
# downside-risk-adjusted 20d trend: lagged 20d return divided by downside deviation of last 20 daily returns
f=(p.shift(1)/p.shift(21)-1)/(r.shift(1).where(r.shift(1)<0,0).rolling(20,min_periods=15).std()*np.sqrt(20)+1e-8)
f=f.replace([np.inf,-np.inf],np.nan)
fr=p.shift(-1)/p-1
ics=[]; nms=[]; dates=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); nms.append(len(z)); dates.append(dt)
a=np.asarray(ics); print('dates',len(a),'avg_names',np.mean(nms),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12),'hit',np.mean(a>0),'coverage',np.mean(np.array(nms)/15),'turnover',np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)))
for h in [1,3,5,10,20]:
 y=p.shift(-h)/p-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.array(aa); print('h',h,'ic',aa.mean(),'icir',aa.mean()/(aa.std(ddof=1)+1e-12),'n',len(aa))
# regimes by calendar
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 aa=[v for d,v in zip(dates,a) if lo<=str(d.year)<=hi]
 print('regime',lo,hi,'n',len(aa),'icir',np.mean(aa)/(np.std(aa,ddof=1)+1e-12) if len(aa)>1 else np.nan,'ic',np.mean(aa) if aa else np.nan)
