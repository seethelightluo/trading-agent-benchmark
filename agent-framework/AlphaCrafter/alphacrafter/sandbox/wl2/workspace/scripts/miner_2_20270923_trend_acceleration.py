import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 x=pd.read_csv(p,parse_dates=['date']);x.date=pd.to_datetime(x.date).dt.normalize();return x.set_index('date').sort_index()
D={s:load('../persistent/stock_data/'+s+'.csv') for s in U}; end=pd.Timestamp('2027-09-22'); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});R=C.pct_change()
# Trend acceleration: recent 5d return minus the average daily pace of prior 20d;
# normalize by 20d volatility and lag one completed day.
short=R.rolling(5,min_periods=5).sum(); long=R.rolling(20,min_periods=20).sum(); vol=R.rolling(20,min_periods=20).std().replace(0,np.nan)
F=(short-long/4).div(vol).shift(1);Y=C.shift(-1).div(C)-1
ics=[];ns=[];ds=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:ics.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
a=np.array(ics);print('candidate volatility normalized trend acceleration');print('dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),F.notna().sum().sum()/F.size,F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 q=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'insufficient')
for h in [3,5,10]:
 Yh=C.shift(-h).div(C)-1;q=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Yh.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 q=np.array(q);print('horizon',h,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
