import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(p):
 d=pd.read_csv(p,parse_dates=['date']);d.date=pd.to_datetime(d.date).dt.normalize();return d.set_index('date').sort_index()
D={s:L('../persistent/stock_data/'+s+'.csv') for s in U};dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<='2027-08-25')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});R=C.pct_change(); vol=R.rolling(20,min_periods=15).std()
# volatility-normalized reversal, with VIX shock increasing the reversal horizon.
v=L('../persistent/index_data/VIX.csv').close.reindex(dates).ffill(); shock=(v>v.rolling(60,min_periods=30).median())&(v.pct_change(5)>0)
F=(((-R.rolling(3,min_periods=3).sum()).div(vol)).where(shock,(-R.rolling(1).sum()).div(vol))).shift(1);Y=C.shift(-1).div(C)-1
ics=[];ns=[];ds=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:ics.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
a=np.array(ics);print('dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),F.notna().sum().sum()/F.size,F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 q=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
