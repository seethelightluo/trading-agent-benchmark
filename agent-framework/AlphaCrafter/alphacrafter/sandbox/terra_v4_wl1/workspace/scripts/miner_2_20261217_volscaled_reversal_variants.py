import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d['r']=d.close.pct_change(); d['vol']=d.r.rolling(20,min_periods=15).std().shift(1)
 for lag in [3,5,10,15]: d['f'+str(lag)]=-(d.close.pct_change(lag).shift(1))/(d.vol*np.sqrt(252)+1e-12)
 d['y']=d.close.shift(-1)/d.close-1; R.append(d[['date','y']+[f'f{x}' for x in [3,5,10,15]]].assign(s=s))
p=pd.concat(R)
for lag in [3,5,10,15]:
 a=[]; ns=[]
 for dt,g in p.groupby('date'):
  g=g.dropna(subset=['f'+str(lag),'y'])
  if len(g)>=8: a.append(g['f'+str(lag)].rank().corr(g.y.rank())); ns.append(len(g))
 a=pd.Series(a).dropna(); print(lag,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
