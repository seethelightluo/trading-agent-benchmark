import numpy as np, pandas as pd
import os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def calc():
 px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); px=px.loc[:'2031-09-03']
 # 10d risk-adjusted residual momentum: asset 20d return minus peer median, scaled by 20d vol; smooth 5d
 r=px.pct_change(); mom=px.pct_change(20); peer=mom.sub(mom.median(axis=1),axis=0)
 vol=r.rolling(20).std()*np.sqrt(20)
 sig=-(peer/vol).rolling(5).mean() # contrarian relative shocks
 rows=[]
 for h in [1,5,10,20]:
  fwd=px.shift(-h)/px-1
  vals=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  a=np.array(vals); print('h',h,'dates',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
 # exact 10h details
 fwd=px.shift(-10)/px-1; vals=[]; turns=[]; n=[]
 prev=None
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z))
  rank=sig.loc[dt].rank(pct=True)
  if prev is not None: turns.append(np.nanmean(abs(rank-prev)))
  prev=rank
 a=np.array(vals); print('10d recent365',np.nanmean(a[-365:]),np.nanmean(a[-365:])/np.nanstd(a[-365:],ddof=1),'recent180',np.nanmean(a[-180:]),np.nanmean(a[-180:])/np.nanstd(a[-180:],ddof=1))
 print('coverage',np.mean(n)/15,'avgN',np.mean(n),'turn',np.nanmean(turns),'end',sig.index[-1].date())
if __name__=='__main__': calc()
