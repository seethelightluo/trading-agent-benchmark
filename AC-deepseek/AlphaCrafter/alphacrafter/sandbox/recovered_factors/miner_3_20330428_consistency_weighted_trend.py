import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn);q.date=pd.to_datetime(q.date);d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-04-27']; r=px.pct_change()
# Consistency-weighted trend: 60d risk-adjusted momentum multiplied by fraction of positive daily returns.
mom=px.pct_change(60); vol=r.rolling(40,min_periods=30).std()*np.sqrt(40)
cons=(r.gt(0).rolling(60,min_periods=45).mean()-0.5)*2
sig=(mom/vol*cons).shift(1)
print('candidate consistency_weighted_trend dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1;a=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
