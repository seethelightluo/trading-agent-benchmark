import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-03')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index()
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); fac=-(px/px.shift(3)-1)/(vol+0.005)
def st(a):
 a=np.asarray(a); return len(a),float(np.mean(a)),float(np.mean(a)/(np.std(a,ddof=1)/np.sqrt(len(a)))),float(np.mean(a>0))
for h in [1,5,10,20]:
 out=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],px.shift(-h).loc[dt]/px.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=np.array([x[1] for x in out]); d=pd.to_datetime([x[0] for x in out])
 print(h,'d',st(a),'recent',st(a[-252:]),'2027+',st(a[d>=pd.Timestamp('2027-01-01')]),'2028+',st(a[d>=pd.Timestamp('2028-01-01')]))
print('period',px.index.min().date(),end.date(),'coverage',float(fac.notna().mean().mean()),'avgN',len(U),'turnover',float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2))
