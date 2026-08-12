import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2029-06-13'].set_index('date').sort_index(); D[a]=d.close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v[v.date<='2029-06-13'].set_index('date').sort_index().close
v=v.reindex(px.index).ffill(); vr=v.pct_change()
# Idea: in VIX-stress days, favor assets with recent negative VIX beta (resilient); otherwise use neutral score.
for win in [40,60,90]:
 cov=r.rolling(win,min_periods=win//2).cov(vr); vv=vr.rolling(win,min_periods=win//2).var(); beta=cov.div(vv,axis=0)
 stress=(v>v.rolling(252,min_periods=120).median()).astype(float)
 f=(-beta).where(stress>0).shift(1)
 fr=px.pct_change().shift(-1); vals=[]; ns=[]; turns=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   if i:
    q=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
    if len(q): turns.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
 x=np.array(vals); print('WIN',win,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(len(x)/len(px),4),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4))
 for name,lo,hi in [('pre2027',None,'2027-01-01'),('2027-28','2027-01-01','2029-01-01'),('2029+','2029-01-01',None)]:
  y=np.array([z for d,z in zip([dt for dt in f.index if dt in []],[])])
 # report regime explicitly from indexed list
 pairs=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: pairs.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 for name,lo,hi in [('pre2027',None,'2027-01-01'),('2027-28','2027-01-01','2029-01-01'),('2029+','2029-01-01',None)]:
  y=np.array([q for d,q in pairs if (lo is None or d>=pd.Timestamp(lo)) and (hi is None or d<pd.Timestamp(hi))])
  print(' ',name,'dates',len(y),'IC',round(np.nanmean(y),6) if len(y) else None,'ICIR',round(np.nanmean(y)/np.nanstd(y,ddof=1),6) if len(y)>1 else None)
print('instruments',len(D),'rows',len(px),'stressfreq',round(stress.mean(),4),'last',px.index[-1])
