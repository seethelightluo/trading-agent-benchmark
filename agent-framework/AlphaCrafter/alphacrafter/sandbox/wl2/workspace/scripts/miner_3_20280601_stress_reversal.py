import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  d['date']=pd.to_datetime(d['date']);P[s]=d.set_index('date')['close'].sort_index()
px=pd.concat(P,axis=1).sort_index().ffill(); r=px.pct_change(); mr=r.mean(axis=1)
# Stress-conditioned short-term reversal: recent 1d loss is favored only when
# cross-asset market volatility is elevated; lagged to prevent look-ahead.
stress=mr.rolling(20,min_periods=15).std()/mr.rolling(120,min_periods=60).std()
f=(-r.mul(stress,axis=0)).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,3,5,10]:
 vals=[];ns=[]
 fr=px.pct_change(h).shift(-h)
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);ns.append(len(a))
 x=pd.Series(vals);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
