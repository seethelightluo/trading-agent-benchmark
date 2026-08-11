import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-04-07'); dates=D['SPX'].index[(D['SPX'].index>='2020-03-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); mom=C/C.shift(20)-1
# Smoothed cross-sectional rank of medium momentum, reducing noisy rank flips.
F=mom.rank(axis=1,pct=True).rolling(5,min_periods=4).mean().shift(1); y=C.shift(-1).div(C)-1
def run(Y):
 a=[];ds=[];ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a);return a,ds,ns
a,ds,ns=run(y);print('factor smoothed_rank_momentum_20d dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.diff().abs().mean(axis=1).mean(),4),'end',end.date())
for h in [3,5,10]:
 aa,_,_=run(C.shift(-h).div(C)-1);print('H',h,'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6),'n',len(aa))
