import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);d=d[d.date<=pd.Timestamp('2029-05-16')].set_index('date').sort_index();D[a]=d.close
px=pd.concat(D,axis=1).sort_index();r=px.pct_change()
# Relative reversal: recent asset return minus contemporaneous cross-asset median, then invert and risk-scale.
rr=r-r.median(axis=1).values[:,None]
vol=r.rolling(20,min_periods=20).std()
f=(-rr.rolling(5,min_periods=5).sum()/vol).shift(1)
print('instruments',len(D),'rows',len(px))
for h in [1,3,5,10]:
 fr=px.pct_change(h).shift(-h); vals=[];ns=[];turn=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
   if i:
    q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
    if len(q)>=8:turn.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
 x=np.array(vals);print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turn),4))
fr=px.pct_change(1).shift(-1);vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for n,lo,hi in [('pre27','1900','2027-01-01'),('27-28','2027-01-01','2029-01-01'),('29+','2029-01-01','2100')]:
 x=np.array([v for d,v in vals if pd.Timestamp(lo)<=d<pd.Timestamp(hi)]);print(n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
f.to_csv('scripts/miner_1_20290517_relative_reversal_signal.csv')
