import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in keep:
  x=pd.read_csv(f); x.date=pd.to_datetime(x.date); d[s]=x.set_index('date')
px=pd.DataFrame({s:x.close for s,x in d.items()}).sort_index().loc[:'2033-02-02']
r=px.pct_change()
# Recovery efficiency: recent 10d gain relative to worst peak-to-trough loss in preceding 20d; lagged.
# Rewards assets recovering efficiently from drawdowns, while avoiding raw momentum scale.
peak=px.rolling(20,min_periods=15).max(); dd=px/peak-1
mdd=(-dd.rolling(20,min_periods=15).min()).replace(0,np.nan)
f=(r.rolling(10,min_periods=8).sum()/mdd).shift(1)
print('candidate recovery_efficiency_10_20; dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,a,b in [('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-02-02')]:
 vals=[];fr=px.shift(-10)/px-1
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(vals);print('REG10',label,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else 'NA')
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
