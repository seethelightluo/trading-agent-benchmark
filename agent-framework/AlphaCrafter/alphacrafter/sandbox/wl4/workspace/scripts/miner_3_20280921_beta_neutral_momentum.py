import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2028-09-20')
P={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.DataFrame(P).sort_index(); p=p.loc[:cut]; r=p.pct_change(); bench=r.mean(axis=1)
# Relative momentum after removing each asset's rolling 60d beta exposure to equal-weight benchmark. Lagged one day.
cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
br=(1+bench).rolling(20,min_periods=20).apply(np.prod,raw=True)-1
ar=(1+r).rolling(20,min_periods=20).apply(np.prod,raw=True)-1
f=(ar-beta.mul(br,axis=0)).shift(1)
def calc(t,h):
 y=(1+r.loc[t:].iloc[1:h+1]).prod()-1; z=pd.concat([f.loc[t],y],axis=1).dropna()
 if len(z)<8:return np.nan,0
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)
for h in [1,5,10,20]:
 a=[];ns=[]
 for t in f.index:
  x,n=calc(t,h)
  if np.isfinite(x):a.append(x);ns.append(n)
 a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028-09-20')]:
 a=[]
 for t in f.loc[lo:hi].index:
  x,n=calc(t,10)
  if np.isfinite(x):a.append(x)
 a=np.array(a);print('REG',lo,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
R=f.rank(axis=1,pct=True); tv=[]
for i in range(1,len(R)):
 z=pd.concat([R.iloc[i-1],R.iloc[i]],axis=1).dropna()
 if len(z)>=8:tv.append(abs(z.iloc[:,0]-z.iloc[:,1]).mean())
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(np.mean(tv),5),'min_dates',f.index.min().date(),'max_date',f.index.max().date())
