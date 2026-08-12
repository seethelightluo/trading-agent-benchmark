import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2029-05-30'].set_index('date').sort_index(); D[a]=d.close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
# Trend curvature: recent 5d return minus average daily 20d return, scaled by 20d risk; contrarian
f=-(r.rolling(5,min_periods=5).sum()-r.rolling(20,min_periods=20).sum()/4)/(r.rolling(20,min_periods=20).std()*np.sqrt(20))
f=f.shift(1)
fr=px.pct_change(1).shift(-1); vals=[]; ns=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)); ns.append(len(z))
  if i:
   q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna(); turns.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
x=np.array([v for d,v in vals]); print('dates',len(x),'avgN',round(np.mean(ns),2),'instruments',len(D),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4))
for name,lo,hi in [('pre2027',None,'2027-01-01'),('2027-28','2027-01-01','2029-01-01'),('2029','2029-01-01',None)]:
 y=np.array([v for d,v in vals if (lo is None or d>=pd.Timestamp(lo)) and (hi is None or d<pd.Timestamp(hi))]); print(name,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6))
for h in [3,5,10]:
 fr=px.pct_change(h).shift(-h); z=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 y=np.array(z);print('h',h,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
f.to_csv('scripts/miner_3_20290531_curvature_reversal_signal.csv')
