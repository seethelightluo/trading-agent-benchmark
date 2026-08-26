import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-23')
P={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); up=r.clip(lower=0).rolling(20,min_periods=15).mean(); dn=(-r.clip(upper=0)).rolling(20,min_periods=15).mean(); fac=up/(dn+1e-8)
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==10:
  for label,sel in [('all',np.ones(len(a),bool)),('recent252',np.arange(len(a)>=len(a)-252),)]: pass
  for label,sel in [('recent252',np.arange(len(a))>=len(a)-252),('prior',np.arange(len(a))<len(a)-252)]:
   b=a[sel]; print(label,'N',len(b),'IC',round(b.mean(),6),'ICIR',round(b.mean()/b.std(ddof=1),6),'hit',round((b>0).mean(),4))
print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
