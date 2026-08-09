import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-06-22']; r=px.pct_change()
# Trend-persistence quality: medium-term return rewarded only when daily direction is persistent.
# This avoids a single jump dominating raw momentum; lag one completed day.
ret20=px.pct_change(20)
persist=r.rolling(20,min_periods=16).apply(lambda x: np.mean(x>0),raw=True)
sig=(ret20 * (0.5 + persist)).shift(1)
print('candidate trend_persistence_20 dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for label,lo,hi in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-06-22')]:
  aa=np.array([v for v,dt in zip(vals,ds) if str(dt)>=lo and str(dt)<=hi]) if h==1 else np.array([])
  if len(aa): print(' regime',label,'n',len(aa),'IC',round(aa.mean(),6),'ICIR',round(aa.mean()/aa.std(ddof=1),6))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
