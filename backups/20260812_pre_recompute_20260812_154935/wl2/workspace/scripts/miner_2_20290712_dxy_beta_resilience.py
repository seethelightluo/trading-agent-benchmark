import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2029-07-11'].set_index('date').sort_index(); D[a]=d.close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
x=pd.read_csv('../persistent/index_data/DXY.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<='2029-07-11'].set_index('date').sort_index().close.reindex(px.index).ffill(); xr=x.pct_change()
# DXY-resilience: assets with negative rolling DXY beta are favored when DXY is above its long-term median.
for win in [40,60,90]:
 beta=r.rolling(win,min_periods=win//2).cov(xr).div(xr.rolling(win,min_periods=win//2).var(),axis=0)
 stress=(x>x.rolling(252,min_periods=120).median())
 f=(-beta).where(stress).shift(1); fr=px.pct_change().shift(-1); pairs=[]; turns=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   pairs.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
   if i:
    q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
    if len(q): turns.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
 vals=np.array([q for _,q in pairs]); print('WIN',win,'dates',len(vals),'avgN',round(np.mean([len(pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()) for d,_ in pairs]),2),'coverage',round(len(vals)/len(px),4),'IC',round(np.nanmean(vals),6),'ICIR',round(np.nanmean(vals)/np.nanstd(vals,ddof=1),6),'hit',round(np.mean(vals>0),4),'turn',round(np.nanmean(turns),4))
 for name,lo,hi in [('pre2027',None,'2027-01-01'),('2027-28','2027-01-01','2029-01-01'),('2029+','2029-01-01',None)]:
  y=np.array([q for d,q in pairs if (lo is None or d>=pd.Timestamp(lo)) and (hi is None or d<pd.Timestamp(hi))]); print(name,len(y),round(np.nanmean(y),6) if len(y) else None,round(np.nanmean(y)/np.nanstd(y,ddof=1),6) if len(y)>1 else None)
print('instruments',len(D),'rows',len(px),'stressfreq',round(stress.mean(),4),'last',px.index[-1])
