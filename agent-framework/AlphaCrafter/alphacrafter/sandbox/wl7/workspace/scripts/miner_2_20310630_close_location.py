import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs=[]
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None: fs.append(d[['date','close','high','low']].set_index('date').rename(columns={'close':s+'_c','high':s+'_h','low':s+'_l'}))
x=pd.concat(fs,axis=1).sort_index().ffill(); c=x[[s+'_c' for s in U]]; c.columns=U
# prior day's close location in its daily range, cross-sectionally demeaned; low close predicts rebound
raw=pd.DataFrame(index=x.index,columns=U,dtype=float)
for s in U: raw[s]=(x[s+'_c']-x[s+'_l'])/(x[s+'_h']-x[s+'_l']).replace(0,np.nan)
f=(raw.sub(raw.median(axis=1),axis=0)).shift(1)
for h in [1,5,10]:
 fr=c.shift(-h)/c-1; qs=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(qs).dropna();print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
if True: raw.to_csv('scripts/miner_2_20310630_close_location_signal.csv')
