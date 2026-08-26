import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
close=pd.DataFrame(p).sort_index().ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(close.index).ffill()
r=close.pct_change(); vol=r.rolling(20,min_periods=20).std()*np.sqrt(252)
vp=v.shift(1).rolling(252,min_periods=60).rank(pct=True)
base=r.rolling(10,min_periods=10).sum().shift(1)/(vol.shift(1)+1e-12)
# calm trend, stressed reversal; all inputs lagged one session
f=base.where(vp<0.65,-base).rolling(3,min_periods=3).mean()
fr=close.shift(-10)/close-1
print('through',close.index.max().date(),'dates',len(close),'assets',len(close.columns))
for h in [1,5,10,20]:
 y=close.shift(-h)/close-1; vals=[]; ns=[]
 for dt in close.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(vals).dropna(); print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for label,mask in [('pre2030',close.index<'2030-01-01'),('post2030',close.index>='2030-01-01'),('recent365',close.index>=close.index.max()-pd.Timedelta(days=365))]:
 vals=[]
 for dt in close.index[mask]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); print(label,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover_proxy',f.rank(pct=True).diff().abs().mean().mean())
