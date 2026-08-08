import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in keep:
  x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); d[s]=x.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2032-12-15']; r=px.pct_change()
# Smoothed acceleration: average of 5 lagged observations of 10d return acceleration.
raw=r.rolling(10,min_periods=10).sum()-r.rolling(20,min_periods=20).sum()+0 # R10(t)-R10(t-10)
f=raw.rolling(5,min_periods=5).mean().shift(1)
print('candidate smoothed_return_acceleration_10_5; dates through',px.index.max().date(),'universe',len(d))
for h in [1,5,10,20]:
 vals=[]; ns=[]
 fr=px.shift(-h)/px-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
# regimes, same H10
fr=px.shift(-10)/px-1
for label,a,b in [('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-32','2031','2032-12-15')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.asarray(vals);print('REG10',label,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
