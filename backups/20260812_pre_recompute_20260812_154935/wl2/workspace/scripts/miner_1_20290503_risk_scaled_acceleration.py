import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<=pd.Timestamp('2029-05-02')].set_index('date').sort_index(); D[a]=d.close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
# acceleration: recent 10d return relative to prior 30d average, scaled by trailing 20d risk
acc=r.rolling(10,min_periods=10).sum()-r.rolling(30,min_periods=30).sum()/3
vol=r.rolling(20,min_periods=20).std()*np.sqrt(20)
f=(acc/vol.replace(0,np.nan)).shift(1)
print('instruments',len(D),'rows',len(px),'range',px.index.min(),px.index.max())
for h in [1,3,5,10]:
 fr=px.pct_change(h).shift(-h); vals=[]; ns=[]; turns=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna(); n=len(z)
  if n>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(n)
   if i:
    q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
    if len(q)>=8: turns.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
 x=np.asarray(vals); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4))
fr=px.pct_change(1).shift(-1); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for name,sel in [('pre2026',lambda d:d<pd.Timestamp('2026-01-01')),('2026-27',lambda d:pd.Timestamp('2026-01-01')<=d<pd.Timestamp('2028-01-01')),('2028+',lambda d:d>=pd.Timestamp('2028-01-01'))]:
 x=np.array([v for d,v in vals if sel(d)]); print(name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
