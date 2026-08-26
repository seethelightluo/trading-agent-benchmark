import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
pd0=pd.DataFrame(p).sort_index().ffill(); ret=pd0.pct_change();
# Breadth-conditioned short-term reversal: lagged 5d asset reversal, activated with stronger weight when market breadth is weak.
bread=(ret.shift(1)>0).mean(axis=1); market=ret.mean(axis=1).rolling(5,min_periods=5).sum().shift(1)
# continuous condition: reversal in weak breadth, trend in strong breadth
condition=(0.5-bread).clip(-.5,.5)*2
f=(-ret.rolling(5,min_periods=5).sum().shift(1)).mul(condition,axis=0).rolling(3,min_periods=3).mean()
print('through',pd0.index.max().date(),'dates',len(pd0),'assets',len(pd0.columns))
for h in [1,5,10,20]:
 y=pd0.shift(-h)/pd0-1; vals=[];ns=[]
 for dt in pd0.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(vals).dropna();print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for label,mask in [('pre2030',pd0.index<'2030-01-01'),('post2030',pd0.index>='2030-01-01'),('recent365',pd0.index>=pd0.index.max()-pd.Timedelta(days=365))]:
 vals=[]
 for dt in pd0.index[mask]:
  z=pd.concat([f.loc[dt],(pd0.shift(-10)/pd0-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna();print(label,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover_proxy',f.rank(pct=True).diff().abs().mean().mean())
