import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=2600)
 if d is not None:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index()
P=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index().ffill();R=np.log(P).diff()
# Recent shock reversal, selectively emphasize moves occurring in high own volatility
z=R.iloc[-1:] # placeholder
rv=R.rolling(20,min_periods=20).std(); f=(-R/(rv+1e-8)*(1+R.abs()/rv)).shift(1)
def ev(Y,sl=slice(None)):
 a=[];ns=[]
 for dt in f.loc[sl].index:
  q=pd.concat([f.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   x=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(x):a.append(x);ns.append(len(q))
 a=np.array(a);return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
for h in [1,3,5,10]:print('h',h,ev(np.log(P).shift(-h)-np.log(P)))
print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),5),'dates',len(P),'instruments',len(D))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]:print(n,ev(np.log(P).shift(-1)-np.log(P),s))
