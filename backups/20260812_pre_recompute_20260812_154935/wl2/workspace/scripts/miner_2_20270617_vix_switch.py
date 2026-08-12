import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p,parse_dates=['date']); d['date']=pd.to_datetime(d['date']).dt.normalize(); return d.set_index('date').sort_index()
D={s:load('../persistent/stock_data/'+s+'.csv') for s in U}; vix=load('../persistent/index_data/VIX.csv')['close']
end=pd.Timestamp('2027-06-16'); dates=D['SPX'].index[(D['SPX'].index>=pd.Timestamp('2020-04-01'))&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change(); vx=vix.reindex(dates).ffill()
reg=(vx>vx.rolling(60,min_periods=30).median()).astype(float)
rev=-R.rolling(5,min_periods=5).sum(); mom=R.rolling(20,min_periods=15).sum()
F=rev.mul(reg,axis=0)+mom.mul(1-reg,axis=0); F=F.shift(1); y=C.shift(-1).div(C)-1
def calc(Y):
 a=[];ds=[];ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
 return np.array(a),ds,ns
a,ds,ns=calc(y);print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4));print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for h in [3,5,10]:
 aa,_,_=calc(C.shift(-h).div(C)-1);print('h',h,'n',len(aa),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6))
print('reg elevated fraction',round(reg.mean(),3))
