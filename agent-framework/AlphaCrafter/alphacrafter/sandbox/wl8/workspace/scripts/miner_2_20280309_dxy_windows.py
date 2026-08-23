import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2028-03-08')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index();r=px.pct_change();x=pd.read_csv('../persistent/index_data/DXY.csv');x.date=pd.to_datetime(x.date);d=x[x.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill();dr=d.pct_change();base=r.rolling(3,min_periods=3).sum().shift(1);fw=px.shift(-1)/px-1
for w in [3,5,10,20]:
 sig=base.where(dr.rolling(w,min_periods=w).sum().shift(1)<0,-base); a=[];ds=[];ns=[]
 for dt in px.index:
  g=pd.DataFrame({'x':sig.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(g)>=8 and g.x.nunique()>1:a.append(spearmanr(g.x,g.y).statistic);ds.append(dt);ns.append(len(g))
 a=np.array(a);print('window',w,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'recent',np.mean([a[i] for i,z in enumerate(ds) if z>=END-pd.Timedelta(days=180)]))
