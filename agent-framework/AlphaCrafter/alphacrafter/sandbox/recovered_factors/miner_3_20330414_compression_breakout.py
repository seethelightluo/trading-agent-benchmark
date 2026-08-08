import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-04-13']; ret=px.pct_change()
# One interpretable idea: trend breakout after volatility compression.
# 10d trend is scaled by 20d volatility and rewarded when 5d vol is below 20d vol.
v5=ret.rolling(5,min_periods=4).std(); v20=ret.rolling(20,min_periods=15).std()
sig=(ret.rolling(10,min_periods=8).sum()/v20*(v20/(v5+1e-12)).clip(.25,4)).shift(1)
print('candidate compression_breakout_10_5_20 dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
fr=px.shift(-1)/px-1
for label,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-04-13')]:
 vals=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a1=np.asarray(vals); print('REG1',label,'dates',len(a1),'IC',round(a1.mean(),6),'ICIR',round(a1.mean()/a1.std(ddof=1),6) if len(a1)>1 else None)
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
