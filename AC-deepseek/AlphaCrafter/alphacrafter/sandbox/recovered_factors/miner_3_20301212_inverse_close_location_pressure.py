import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
# Prior completed-session close-location pressure: average signed position of close within daily range.
# Positive values indicate buying pressure; use contrarian sign as candidate.
cl=[]
for a,d in D.items():
 rng=(d.high-d.low).replace(0,np.nan)
 cl.append(((2*d.close-d.high-d.low)/rng).rename(a))
clv=pd.concat(cl,axis=1).sort_index(); sig=-clv.shift(1).rolling(5,min_periods=3).mean()
prices=pd.DataFrame({a:d.close for a,d in D.items()}).sort_index()
print('candidate=inverse_close_location_pressure_5obs; dates',len(prices),'instruments',len(assets),'coverage',round(sig.notna().sum().sum()/sig.size,6))
for h in [1,5,10,20]:
 f=prices.shift(-h)/prices-1; arr=[]; ns=[]
 for dt in prices.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):arr.append(q);ns.append(len(z))
 arr=np.array(arr);print('H',h,'dates',len(arr),'meanN',round(np.mean(ns),2),'IC',round(arr.mean(),6),'ICIR',round(arr.mean()/arr.std(ddof=1),6),'hit',round(np.mean(arr>0),4))
for label,mask in [('2020-23',prices.index<'2024-01-01'),('2024-27',(prices.index>='2024-01-01')&(prices.index<'2028-01-01')),('latest120',prices.index>=prices.index[-120])]:
 f=prices.shift(-1)/prices-1;arr=[]
 for dt in prices.index[mask]:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:arr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 arr=np.array(arr);print(label,'dates',len(arr),'IC',round(arr.mean(),6),'ICIR',round(arr.mean()/arr.std(ddof=1),6))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lag in [1,3,5,10]:
 z=pd.concat([sig.stack().rename('a'),sig.shift(lag).stack().rename('b')],axis=1).dropna();print('decay',lag,round(spearmanr(z.a,z.b).statistic,6))
# explicit library proxy audit, aligned cells, admissible factors reconstructed from definitions
r=prices.pct_change(); lib={
'ravmom20':(prices.shift(1)/prices.shift(21)-1)/r.rolling(20,min_periods=15).std().shift(1),
'volnormrev5':-(prices.shift(1)/prices.shift(6)-1)/r.rolling(5,min_periods=4).std().shift(1),
'trendconsistency':((prices.shift(1)/prices.shift(21)-1)/r.rolling(20,min_periods=15).std().shift(1))*((r.shift(1).rolling(20,min_periods=15).gt(0).mean())-.5),
'overnightgap':-pd.concat([(d.open/d.close.shift(1)-1).rename(a) for a,d in D.items()],axis=1).sort_index().shift(1).rolling(3,min_periods=2).mean(),
'relvol':pd.DataFrame({a:d.volume for a,d in D.items()}).sort_index().shift(1).apply(lambda x: np.log(x/x.rolling(20,min_periods=10).mean()))}
mx=0;bn=''
for n,s in lib.items():
 z=pd.concat([sig.stack().rename('a'),s.stack().rename('b')],axis=1).dropna();q=abs(spearmanr(z.a,z.b).statistic)
 print('corr',n,round(q,6),'cells',len(z))
 if q>mx:mx=q;bn=n
print('MAX_CORR',round(mx,6),bn)
