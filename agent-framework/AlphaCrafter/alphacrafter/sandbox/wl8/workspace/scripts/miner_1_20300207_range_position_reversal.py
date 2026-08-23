import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-02-06')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close.astype(float)
prices=pd.concat(D,axis=1).sort_index().loc[:END]; rets=prices.pct_change(); hi=prices.rolling(120,min_periods=60).max(); lo=prices.rolling(120,min_periods=60).min(); rng=(hi-lo).replace(0,np.nan); pos=(prices-lo)/rng; vol=rets.rolling(20,min_periods=10).std()
sig=(-(pos-0.5)/(vol*np.sqrt(20))).shift(1).clip(-10,10); fwd=prices.shift(-10)/prices-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic): rows.append((dt,ic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('candidate range_position_volscaled_reversal'); print('dates',len(R),'period',R.index.min(),R.index.max(),'avg_n',R.n.mean(),'coverage',len(R)/(len(sig)-10))
if len(R):
 print('IC %.6f ICIR %.6f hit %.4f turnover_proxy %.6f'%(R.ic.mean(),R.ic.mean()/R.ic.std(ddof=1),(R.ic>0).mean(),sig.rank(axis=1).diff().abs().mean().mean()/14))
 for name,mask in [('2026',R.index.year==2026),('2027-28',R.index.year.isin([2027,2028])),('2029',R.index.year==2029),('recent360',R.index>=R.index.max()-pd.Timedelta(days=360)),('recent180',R.index>=R.index.max()-pd.Timedelta(days=180))]:
  q=R[mask]; print(name,len(q),('IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))) if len(q)>2 else 'NA')
 for h in [1,5,10,20]:
  fw=prices.shift(-h)/prices-1; rr=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  rr=pd.Series(rr).dropna(); print('decay',h,'IC %.6f ICIR %.6f'%(rr.mean(),rr.mean()/rr.std(ddof=1)))
