import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); D[a]=x.set_index('date').close
px=pd.concat(D,axis=1).sort_index().ffill()
r=px.pct_change()
# cross-sectional dispersion: mean abs deviation from daily median, and its 60d percentile
med=r.median(axis=1); disp=r.sub(med,axis=0).abs().mean(axis=1)
dpct=disp.rolling(120,min_periods=60).rank(pct=True)
# Conditional short-horizon reversal, scaled by own vol; condition on high dispersion
for name, cond in [('highdisp',dpct>=.7),('lowdisp',dpct<=.3),('all',pd.Series(True,index=px.index))]:
 for L in [3,5,10]:
  vol=r.rolling(20,min_periods=15).std()
  f=-(px.pct_change(L)/vol.replace(0,np.nan)).where(cond, np.nan)
  print('\n',name,L)
  for H in [1,5,10,20]:
   fr=px.pct_change(H).shift(-H)
   ics=[]; ns=[]
   for dt in px.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   q=np.array(ics); print('H',H,'dates',len(q),'N',round(np.mean(ns),2) if ns else 0,'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6) if len(q)>1 else 0,'hit',round(np.mean(q>0),3) if len(q) else 0)
 print('coverage',f.notna().stack().mean())
