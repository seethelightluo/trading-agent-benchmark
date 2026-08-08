import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
prices=pd.DataFrame({a:d.close for a,d in D.items()}).sort_index(); rets=prices.pct_change()
# Range-expansion directional pressure: recent close-location pressure weighted by abnormal true range.
cl=[]; tr=[]
for a,d in D.items():
 rng=(d.high-d.low).replace(0,np.nan)
 cl.append(((2*d.close-d.high-d.low)/rng).rename(a))
 tr.append(((d.high-d.low)/d.close.shift(1)).rename(a))
cl=pd.concat(cl,axis=1).sort_index(); tr=pd.concat(tr,axis=1).sort_index()
sig=(cl.shift(1).rolling(5,min_periods=3).mean()* (tr.shift(1)/tr.shift(1).rolling(20,min_periods=10).mean())).replace([np.inf,-np.inf],np.nan)
print('candidate=range_expansion_directional_pressure; source_dates',len(prices),'assets',len(assets),'cell_coverage',round(sig.notna().sum().sum()/sig.size,6))
def calc(h, dates):
 f=prices.shift(-h)/prices-1; arr=[]; ns=[]
 for dt in dates:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):arr.append(q);ns.append(len(z))
 a=np.array(arr); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
for h in [1,5,10,20]: print('H',h,'dates %.0f meanN %.2f IC %.6f ICIR %.6f hit %.4f'%calc(h,prices.index))
for name,mask in [('2020-23',prices.index<'2024-01-01'),('2024-27',(prices.index>='2024-01-01')&(prices.index<'2028-01-01')),('2028+',prices.index>='2028-01-01'),('latest120',prices.index>=prices.index[-120])]:
 print(name,'H1 dates %.0f meanN %.2f IC %.6f ICIR %.6f hit %.4f'%calc(1,prices.index[mask]))
print('rank_turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lag in [1,3,5,10]: print('decay',lag,round(spearmanr(sig.stack(),sig.shift(lag).stack(),nan_policy='omit').statistic,6))
# full auditable proxy library
r=prices.pct_change(); lib={'ravmom20':(prices.shift(1)/prices.shift(21)-1)/r.rolling(20,min_periods=15).std().shift(1),'volnormrev5':-(prices.shift(1)/prices.shift(6)-1)/r.rolling(5,min_periods=4).std().shift(1),'trendconsistency':((prices.shift(1)/prices.shift(21)-1)/r.rolling(20,min_periods=15).std().shift(1))*(r.shift(1).gt(0).rolling(20,min_periods=15).mean()-.5),'overnightgap':-pd.concat([(d.open/d.close.shift(1)-1).rename(a) for a,d in D.items()],axis=1).sort_index().shift(1).rolling(3,min_periods=2).mean(),'relvol':pd.DataFrame({a:d.volume for a,d in D.items()}).sort_index().shift(1).apply(lambda x:np.log(x/x.rolling(20,min_periods=10).mean()))}
mx=0
for n,s in lib.items():
 z=pd.concat([sig.stack().rename('a'),s.stack().rename('b')],axis=1).dropna();q=abs(spearmanr(z.a,z.b).statistic);print('corr',n,round(q,6),'cells',len(z));mx=max(mx,q)
print('MAX_CORR',round(mx,6))
