import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-03')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index()
r=px.pct_change()
ret10=px/px.shift(10)-1
# Trend quality: directional 10d return rewarded when path is smooth, penalized by downside volatility.
down=r.where(r<0,0).rolling(20,min_periods=15).std()
quality=ret10/(down+0.005)
fac=quality.rank(axis=1,pct=True)
records={}
for h in [1,5,10,20]:
 out=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],px.shift(-h).loc[dt]/px.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 records[h]=out
 def stat(x):
  x=np.asarray(x); return (len(x),float(np.mean(x)),float(np.mean(x)/(np.std(x,ddof=1)/np.sqrt(len(x)))) if len(x)>1 else np.nan,float(np.mean(x>0)))
 a=np.array([x[2] for x in out]); d=pd.to_datetime([x[0] for x in out]); ns=np.array([x[1] for x in out])
 print(h,'d',stat(a),'recent252',stat(a[-252:]),'2027+',stat(a[d>=pd.Timestamp('2027-01-01')]),'2028+',stat(a[d>=pd.Timestamp('2028-01-01')]))
print('period',px.index.min().date(),end.date(),'coverage',float(fac.notna().mean().mean()),'avgN',np.mean([x[1] for x in records[5]]))
print('turnover',float(fac.diff().abs().mean(axis=1).mean()*2))
