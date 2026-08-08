import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in A:
  q=pd.read_csv(f); q['date']=pd.to_datetime(q.date); d[s]=q.set_index('date').close
p=pd.DataFrame(d).sort_index().loc[:'2032-12-15']; r=p.pct_change()
# Volatility-normalized medium momentum, lagged one completed day.
# 20d return divided by 20d realized vol, with a 5d fast return overlay to avoid stale ranks.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(0.7*r.rolling(20,min_periods=15).sum()/vol + 0.3*r.rolling(5,min_periods=5).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(5))).shift(1)
print('candidate=blended_vol_scaled_momentum_5_20; dates',len(p),'assets',len(A),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ic=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.array(ic); print('H',h,'dates',len(x),'meanN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6),'hit',round((x>0).mean(),4))
# 10d regime and decay
for lab,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-32','2031','2032-12-15')]:
 y=p.shift(-10)/p-1; x=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x); print('REG10',lab,len(x),round(x.mean(),6),round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
