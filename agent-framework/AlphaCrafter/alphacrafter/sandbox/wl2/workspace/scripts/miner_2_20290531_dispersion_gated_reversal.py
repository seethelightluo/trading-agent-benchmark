import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2029-05-30'].set_index('date').sort_index(); D[a]=d.close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change(); csdisp=r.std(axis=1).rolling(20,min_periods=15).mean(); threshold=csdisp.rolling(252,min_periods=120).median(); gate=csdisp>threshold
for lb in [3,5,10]:
 vol=r.rolling(20,min_periods=15).std(); base=-r.rolling(lb,min_periods=lb).sum()/vol; f=base.where(gate, np.nan).shift(1)
 print('LOOKBACK',lb)
 for h in [1,3,5,10]:
  fr=px.pct_change(h).shift(-h); vals=[]; ns=[]; turns=[]
  for i,dt in enumerate(f.index):
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
    vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    if i:
     q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna(); turns.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
  x=np.array(vals); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4))
 fr=px.pct_change().shift(-1); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 for name,lo,hi in [('pre2027',None,'2027-01-01'),('2027-28','2027-01-01','2029-01-01'),('2029+', '2029-01-01',None)]:
  x=np.array([v for d,v in vals if (lo is None or d>=pd.Timestamp(lo)) and (hi is None or d<pd.Timestamp(hi))]); print(name,'dates',len(x),'IC',round(np.nanmean(x),6) if len(x) else None,'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6) if len(x)>1 else None)
print('instruments',len(D),'rows',len(px),'gatefreq',round(gate.mean(),4),'last',px.index[-1])
f.to_csv('scripts/miner_2_20290531_dispersion_gated_reversal_signal.csv')
