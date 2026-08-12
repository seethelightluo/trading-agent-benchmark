import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change()
# Market-dispersion beta: sensitivity to contemporaneous cross-asset dispersion, lagged into signal
market=r.mean(axis=1); disp=r.std(axis=1)
fac=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U:
 fac[s]=-(r[s].rolling(60,min_periods=40).cov(disp)/disp.rolling(60,min_periods=40).var())
for h in [1,5,10]:
 a=[];ns=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q));ds.append(px.index[i])
 a=np.array(a); print('dispersion_beta',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0))
 for y in [2020,2021,2022,2023,2024,2025,2026]:
  b=a[[d.year==y for d in ds]]
  if len(b): print(y,len(b),round(b.mean(),5),round(b.mean()/b.std(),5))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
rev=-r.rolling(5).sum(); mom=r.rolling(20).sum()
print('corr_momentum',fac.stack().corr(mom.stack()),'corr_reversal',fac.stack().corr(rev.stack()))
